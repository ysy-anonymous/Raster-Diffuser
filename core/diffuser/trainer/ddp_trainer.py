from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from typing import Dict, Optional

from utils.load_utils import build_noise_scheduler_from_config
from core.diffuser.datasets.plane_dataset_embeed import PlanePlanningDataSets
from core.diffuser.networks.diffusion_decoder.helpers.Coarse2FineResHead import interpolate_time


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def build_dataloader_from_dataset_and_config(
    config: Dict,
    dataset: torch.utils.data.Dataset,
    rank: int = 0,
    world_size: int = 1,
):
    batch_size = config["trainer"]["batch_size"]
    num_workers = config["trainer"].get("num_workers", 2)
    pin_memory = config["trainer"].get("pin_memory", True)
    drop_last = config["trainer"].get("drop_last", False)

    sampler = None
    shuffle = True

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )
    return dataloader, sampler


class PlaneDiffusionTrainer:
    def __init__(
        self,
        net: nn.Module,
        dataset: PlanePlanningDataSets,
        config: Dict,
        device: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        dataloader: Optional[DataLoader] = None,
        sampler: Optional[DistributedSampler] = None,
    ):
        self.net = net
        self.config = config.to_dict() if hasattr(config, "to_dict") else config
        self.noise_scheduler = build_noise_scheduler_from_config(self.config)
        self.dataset = dataset

        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        if device is None:
            if torch.cuda.is_available():
                self.device = f"cuda:{rank}" if self.is_distributed else "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # mixed precision settings
        trainer_cfg = self.config["trainer"]
        self.use_amp = trainer_cfg.get("use_amp", True)
        self.amp_dtype_str = trainer_cfg.get("amp_dtype", "fp16").lower()

        if self.amp_dtype_str == "fp16":
            self.amp_dtype = torch.float16
        elif self.amp_dtype_str == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported amp_dtype: {self.amp_dtype_str}")

        # GradScaler is only needed for fp16, not bf16
        self.use_grad_scaler = self.use_amp and (self.amp_dtype == torch.float16) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(device=device, enabled=self.use_grad_scaler)

        target_model = self.net.module if hasattr(self.net, "module") else self.net
        method = getattr(target_model, "set_noise_scheduler", None)
        if callable(method):
            method(self.noise_scheduler)

        self.use_aux = getattr(target_model, "use_aux_loss", None)
        self.T_coarse = getattr(target_model, "T_coarse", None)
        self.T_mid = getattr(target_model, "T_mid", None)
        self.T_fine = getattr(target_model, "T_fine", None)

        self.net.to(self.device)

        if trainer_cfg["optimizer"]["name"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=self.net.parameters(),
                lr=trainer_cfg["optimizer"]["learning_rate"],
                weight_decay=trainer_cfg["optimizer"]["weight_decay"],
            )
        else:
            raise NotImplementedError

        if dataloader is None:
            self.dataloader, self.sampler = build_dataloader_from_dataset_and_config(
                self.config,
                dataset,
                rank=self.rank,
                world_size=self.world_size,
            )
        else:
            self.dataloader = dataloader
            self.sampler = sampler

        self.use_ema = trainer_cfg["use_ema"]
        self.ema = EMAModel(parameters=self.net.parameters(), power=0.75) if self.use_ema else None

        self.lr_scheduler = None

    def prepare_inputs(self, batch):
        action = batch["sample"].to(self.device, dtype=torch.float32, non_blocking=True)
        map_cond = batch["map"].to(self.device, dtype=torch.float32, non_blocking=True)
        env_cond = batch["env"].to(self.device, dtype=torch.float32, non_blocking=True)
        batch_size = action.shape[0]
        return map_cond, env_cond, action, batch_size

    def _forward_loss(self, action, map_cond, env_cond, batch_size):
        noise = torch.randn(action.shape, device=self.device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

        net_out = self.net(noisy_actions, timesteps, map_cond, env_cond)
        if isinstance(net_out, tuple):
            noise_pred = net_out[0]
        else:
            noise_pred = net_out

        if len(noise_pred.shape) == 3:
            loss = nn.functional.mse_loss(noise_pred, noise)
        elif len(noise_pred.shape) == 4:
            err = ((noise_pred - noise[:, None]) ** 2).mean(dim=(2, 3))
            loss = err.min(dim=1).values.mean()
        else:
            raise ValueError(f"Unexpected noise_pred shape: {noise_pred.shape}")

        if self.use_aux:
            if self.T_coarse and self.T_mid:
                aux = net_out[2]
                x0_coarse = interpolate_time(action, self.T_coarse)
                x0_mid = interpolate_time(action, self.T_mid)
                loss_aux_1 = F.mse_loss(aux["traj_coarse"], x0_coarse)
                loss_aux_2 = F.mse_loss(aux["traj_mid"], x0_mid)
                loss_aux_3 = F.mse_loss(aux["traj_fine"], action)
                loss = loss + loss_aux_1 * 0.1 + loss_aux_2 * 0.1 + loss_aux_3

        return loss

    def optimization_step(self, action, map_cond, env_cond, batch_size):
        self.optimizer.zero_grad(set_to_none=True)

        # mixed precision forward
        with torch.amp.autocast(device_type="cuda",
            enabled=self.use_amp and torch.cuda.is_available(),
            dtype=self.amp_dtype,
        ):
            noise = torch.randn(action.shape, device=self.device)
            # loss = self._forward_loss(action, map_cond, env_cond, batch_size)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

            noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)
            net_out = self.net(noisy_actions, timesteps, map_cond, env_cond)            
            noise_pred = net_out[0] if isinstance(net_out, tuple) else net_out

        if len(noise_pred.shape) == 3:
            loss = nn.functional.mse_loss(noise_pred.float(), noise.float())
        elif len(noise_pred.shape) == 4:
            err = ((noise_pred - noise[:, None]) ** 2).mean(dim=(2, 3))
            loss = err.min(dim=1).values.mean()
        else:
            raise ValueError(f"Unexpected noise_pred shape: {noise_pred.shape}")

        if self.use_aux:
            if self.T_coarse and self.T_mid:
                aux = net_out[2]
                x0_coarse = interpolate_time(action, self.T_coarse)
                x0_mid = interpolate_time(action, self.T_mid)
                loss_aux_1 = F.mse_loss(aux["traj_coarse"], x0_coarse)
                loss_aux_2 = F.mse_loss(aux["traj_mid"], x0_mid)
                loss_aux_3 = F.mse_loss(aux["traj_fine"], action)
                loss = loss + loss_aux_1 * 0.1 + loss_aux_2 * 0.1 + loss_aux_3
        
        
        ################# Only Finite Loss Survive #################
        is_finite = torch.tensor(
            1 if torch.isfinite(loss).all() else 0,
            device=self.device,
            dtype=torch.int32
        )
        if self.is_distributed and is_dist_avail_and_initialized():
            dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)
        if is_finite.item() == 0:
            if is_main_process(self.rank):
                print("Non-finite loss detected. Skipping this batch.")
            self.optimizer.zero_grad(set_to_none=True)
            return None
        ############################################################
        
        if self.use_grad_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.use_ema:
            self.ema.step(self.net.parameters())

        return loss.detach()

    def train(
        self,
        num_epochs: int,
        save_ckpt_epoch: int = None,
        save_path: str = "/exhdd/seungyu/diffusion_motion/trained_weights",
    ):
        if save_ckpt_epoch is None:
            save_ckpt_epoch = num_epochs

        self.lr_scheduler = get_scheduler(
            name=self.config["trainer"]["lr_scheduler"]["name"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["trainer"]["lr_scheduler"]["num_warmup_steps"],
            num_training_steps=len(self.dataloader) * num_epochs,
            num_cycles=self.config['trainer']['lr_scheduler']['num_cycles']
        )

        trn_loss = []

        epoch_iter = range(num_epochs)
        if is_main_process(self.rank):
            epoch_iter = tqdm(epoch_iter, desc="Epoch")

        for epoch_idx in epoch_iter:
            self.net.train()

            if self.sampler is not None:
                self.sampler.set_epoch(epoch_idx)

            epoch_loss = []

            batch_iter = self.dataloader
            if is_main_process(self.rank):
                batch_iter = tqdm(self.dataloader, desc="Batch", leave=False)

            for nbatch in batch_iter:
                map_cond, env_cond, action, B = self.prepare_inputs(nbatch)
                loss = self.optimization_step(action, map_cond, env_cond, B)
                
                # If loss is None (Non-finute batch or infinite loss (NaN loss)) skip this loss for reducing.
                if loss is None:
                    continue

                reduced_loss = loss.clone()
                if self.is_distributed and is_dist_avail_and_initialized():
                    reduced_loss = reduce_mean(reduced_loss, self.world_size)

                loss_cpu = reduced_loss.item()
                epoch_loss.append(loss_cpu)

                if is_main_process(self.rank):
                    batch_iter.set_postfix(loss=loss_cpu)

            epoch_mean = float(np.mean(epoch_loss)) if len(epoch_loss) > 0 else 0.0
            trn_loss.append(epoch_mean)

            if is_main_process(self.rank):
                epoch_iter.set_postfix(loss=epoch_mean)

                if (epoch_idx + 1) % save_ckpt_epoch == 0:
                    self.save_checkpoint(path=f"{save_path}/ckpt_ep{epoch_idx + 1}.ckpt")

        return trn_loss

    def save_checkpoint(self, path: str):
        if not is_main_process(self.rank):
            return

        model_to_save = self.net.module if hasattr(self.net, "module") else self.net
        if self.use_ema:
            self.ema.copy_to(model_to_save.parameters())

        ckpt = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        if self.lr_scheduler is not None:
            ckpt["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.use_grad_scaler:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(ckpt, path)