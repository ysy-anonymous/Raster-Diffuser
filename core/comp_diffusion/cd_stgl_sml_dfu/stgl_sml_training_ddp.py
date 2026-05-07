import os
import copy
import numpy as np
import torch
import einops, wandb
import pdb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from core.comp_diffusion.utils.timer import Timer
from core.comp_diffusion.utils.train_utils import get_lr
from core.comp_diffusion.utils.training import cycle, EMA
from core.comp_diffusion.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
import core.comp_diffusion.utils as utils

from core.comp_diffusion.datasets.rrt_map.rrt_map_dataset import rrt_comp_collate_fn
from diffusers.optimization import get_scheduler


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank):
    return rank == 0


class Stgl_Sml_Trainer_DDP(object):
    def __init__(
        self,
        diffusion_model: Stgl_Sml_GauDiffusion_InvDyn_V1,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        horizon=32,
        n_reference=8,
        n_samples=2,
        device='cuda',
        results_folder='./results',
        n_train_steps=30000,
        trainer_dict={},
        rank=0,
        world_size=1,
        local_rank=0,
        use_ddp=False,
        num_workers=6,
        pin_memory=True,
        drop_last=False,
    ):
        super().__init__()

        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.use_ddp = use_ddp and world_size > 1
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.trainer_dict = trainer_dict

        if torch.cuda.is_available():
            self.device = f'cuda:{local_rank}' if self.use_ddp else device
            if self.use_ddp:
                torch.cuda.set_device(local_rank)
        else:
            self.device = 'cpu'

        # AMP settings
        self.use_amp = trainer_dict.get("use_amp", True)
        self.amp_dtype_str = trainer_dict.get("amp_dtype", "fp16").lower()
        if self.amp_dtype_str == "fp16":
            self.amp_dtype = torch.float16
        elif self.amp_dtype_str == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported amp_dtype: {self.amp_dtype_str}")

        self.use_grad_scaler = (
            self.use_amp
            and self.amp_dtype == torch.float16
            and torch.cuda.is_available()
        )
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.use_grad_scaler)

        self.model = diffusion_model.to(self.device)
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=trainer_dict.get("find_unused_parameters", False),
                broadcast_buffers=trainer_dict.get("broadcast_buffers", False),
            )

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.unwrap_model(self.model)).to(self.device)
        self.ema_model.eval()
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.horizon = horizon

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.sampler = None
        shuffle = True
        if self.use_ddp:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=drop_last,
            )
            shuffle = False

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            sampler=self.sampler,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=drop_last,
            collate_fn=rrt_comp_collate_fn,
        )
        print("number of iterations per epoch for constructed dataloader is : ", len(self.dataloader))

        self.dataloader = cycle(self.dataloader)
        self.optimizer = torch.optim.Adam(self.unwrap_model(self.model).parameters(), lr=train_lr)

        self.logdir = results_folder
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.clip_grad_norm = trainer_dict.get("clip_grad_norm", None)
        self.clip_grad_value = trainer_dict.get("clip_grad_value", None)

        self.reset_parameters()
        self.step = 0

    def unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.unwrap_model(self.model).state_dict())

    @torch.no_grad()
    def step_ema(self):
        if not is_main_process(self.rank):
            return
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.unwrap_model(self.model))

    def prepare_inputs(self, batch):
        action = batch['sample'].to(self.device, dtype=torch.float32, non_blocking=True)
        map_cond = batch['map'].to(self.device, dtype=torch.float32, non_blocking=True)

        return action, map_cond

    def train(self, n_train_steps):

        self.lr_scheduler = get_scheduler(
                    name=self.trainer_dict["lr_scheduler"]["name"],
                    optimizer=self.optimizer,
                    num_warmup_steps=self.trainer_dict["lr_scheduler"]["num_warmup_steps"],
                    num_training_steps=n_train_steps,
                    num_cycles=self.trainer_dict['lr_scheduler']['num_cycles']
        )

        if is_main_process(self.rank):
            wandb.init()

        timer = Timer()

        current_epoch = 0
        if self.sampler is not None:
            self.sampler.set_epoch(current_epoch)

        for i_tr in range(n_train_steps):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            last_loss = None
            last_infos = None

            for i_ac in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                obs_trajs, map_cond = self.prepare_inputs(batch)

                base_model = self.unwrap_model(self.model)
                if base_model.tr_cond_type == 'no':
                    cond_st_gl = {}
                else:
                    cond_st_gl = {0: obs_trajs[:, 0, :].squeeze(), self.horizon-1: obs_trajs[:, self.horizon-1, :].squeeze()}

                with torch.amp.autocast(
                    device_type='cuda',
                    enabled=(self.use_amp and torch.cuda.is_available()),
                    dtype=self.amp_dtype,
                ):
                    loss, infos = base_model.loss(
                        x_clean=obs_trajs,
                        cond_st_gl=cond_st_gl,
                        map_cond=map_cond
                    )

                # keep backward on fp32 loss tensor
                loss = loss.float()
                loss = loss / self.gradient_accumulate_every

                if not torch.isfinite(loss):
                    if is_main_process(self.rank):
                        print(f"Non-finite loss detected at step {self.step}. Skipping optimizer step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    last_loss = None
                    last_infos = infos
                    break

                if self.use_grad_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                last_loss = loss
                last_infos = infos

            # only step optimizer if accumulation completed with finite loss
            if last_loss is not None:
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)

                    if self.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(
                            self.unwrap_model(self.model).parameters(),
                            self.clip_grad_value
                        )
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.unwrap_model(self.model).parameters(),
                            self.clip_grad_norm
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(
                            self.unwrap_model(self.model).parameters(),
                            self.clip_grad_value
                        )
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.unwrap_model(self.model).parameters(),
                            self.clip_grad_norm
                        )

                    self.optimizer.step()
                
                # Update LR Scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

            if is_main_process(self.rank) and self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if is_main_process(self.rank) and self.step % self.log_freq == 0 and last_infos is not None:
                loss_item = float("nan") if last_loss is None else last_loss.detach().item()
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in last_infos.items()])
                print(f'{self.step}: {loss_item:8.4f} | {infos_str} | t: {timer():8.4f}')

                metrics = {k: v.detach().item() for k, v in last_infos.items()}
                metrics['train/it'] = self.step
                metrics['train/loss'] = loss_item
                metrics['train/lr'] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)

            self.step += 1

            if self.sampler is not None and (self.step * self.batch_size * self.world_size) % len(self.dataset) < (self.batch_size * self.world_size):
                current_epoch += 1
                self.sampler.set_epoch(current_epoch)

        if is_main_process(self.rank):
            wandb.finish()

    def save(self, epoch):
        if not is_main_process(self.rank):
            return

        data = {
            'step': self.step,
            'model': self.unwrap_model(self.model).state_dict(),
            'ema': self.ema_model.state_dict()
        }

        os.makedirs(self.logdir, exist_ok=True)
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        utils.print_color(f'[ utils/training ] Saved model to {savepath}', c='y')

    def load4resume(self, loadpath):
        data = torch.load(loadpath, map_location=self.device)
        self.unwrap_model(self.model).load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.step = data['step']

    def load(self, epoch):
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath, map_location=self.device)

        self.step = data['step']
        self.unwrap_model(self.model).load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])