# Adapted from pb diffusion github
import os
import copy
import numpy as np
import torch
import einops
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from core.pb_diffusion.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from core.pb_diffusion.utils.timer import Timer
from core.pb_diffusion.utils.train_utils import cycle, EMA
from core.pb_diffusion.utils.train_utils import get_lr
import core.pb_diffusion.networks.diffusion as dmodels
from core.pb_diffusion.utils.train_utils import CosineAnnealingWarmupRestarts
from tqdm import tqdm
import wandb


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank):
    return rank == 0


class TrainerPBDDP(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        results_folder='./results',
        n_samples=2,
        num_render_samples=2,
        clip_grad=False,
        clip_grad_max=False,
        n_train_steps=None,
        trainer_dict={},
        rank=0,
        world_size=1,
        local_rank=0,
        use_ddp=False,
        num_workers=8,
        pin_memory=True,
    ):
        super().__init__()

        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.use_ddp = use_ddp and world_size > 1
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if torch.cuda.is_available():
            self.device = f'cuda:{local_rank}' if self.use_ddp else train_device
            if self.use_ddp:
                torch.cuda.set_device(local_rank)
        else:
            self.device = 'cpu'

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
        self.save_checkpoints = save_checkpoints
        self.step_start_ema = step_start_ema

        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

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
                drop_last=False,
            )
            shuffle = False

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=train_batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )
        self.dataloader = cycle(self.dataloader)

        self.optimizer = torch.optim.Adam(self.unwrap_model(self.model).parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = results_folder
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.num_render_samples = num_render_samples
        self.clip_grad = clip_grad
        self.clip_grad_max = clip_grad_max
        self.lr_warmupDecay = trainer_dict.get('lr_warmupDecay', False)

        if self.lr_warmupDecay:
            assert n_train_steps is not None
            warmup_steps = int(trainer_dict['warmup_steps_pct'] * n_train_steps)
            first_cycle_steps = int(1/trainer_dict['cycle_ratio'] * n_train_steps)
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                first_cycle_steps=first_cycle_steps,
                max_lr=train_lr,
                min_lr=train_lr / 100.0,
                warmup_steps=warmup_steps,
            )
        else:
            self.scheduler = None
        
        self.use_amp = trainer_dict.get("use_amp", True)
        self.amp_dtype_str = trainer_dict.get("amp_dtype", "fp16")

        if self.amp_dtype_str == "fp16":
            self.amp_dtype = torch.float16
        elif self.amp_dtype_str == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError("amp_dtype must be fp16 or bf16")

        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16 and torch.cuda.is_available()

        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.use_grad_scaler)

        self.reset_parameters()
        self.step = 0
        self.debug_mode = False

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
        env_cond = batch['env'].to(self.device, dtype=torch.float32, non_blocking=True)
        return action, map_cond, env_cond

    def train(self, n_train_steps):
        if is_main_process(self.rank):
            wandb.init()

        timer = Timer()

        # for cycling DataLoader with DistributedSampler, set epoch manually
        current_epoch = 0
        if self.sampler is not None:
            self.sampler.set_epoch(current_epoch)

        for step in range(n_train_steps):
            self.model.train()

            self.optimizer.zero_grad(set_to_none=True)

            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = self.prepare_inputs(batch)

                with torch.amp.autocast(
                    device_type='cuda',
                    enabled=self.use_amp,
                    dtype=self.amp_dtype
                ):
                    loss, infos = (
                        self.model.module.loss(*batch)
                        if self.use_ddp
                        else self.model.loss(*batch)
                    )

            # move loss to fp32 region
            loss = loss.float()
            loss = loss / self.gradient_accumulate_every

            if self.use_grad_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.use_grad_scaler:
                self.scaler.unscale_(self.optimizer)

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.unwrap_model(self.model).parameters(),
                        self.clip_grad
                    )
                if self.clip_grad_max:
                    torch.nn.utils.clip_grad_value_(
                        self.unwrap_model(self.model).parameters(),
                        self.clip_grad_max
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.unwrap_model(self.model).parameters(),
                        self.clip_grad
                    )
                if self.clip_grad_max:
                    torch.nn.utils.clip_grad_value_(
                        self.unwrap_model(self.model).parameters(),
                        self.clip_grad_max
                    )

                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if is_main_process(self.rank) and self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if is_main_process(self.rank) and self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss.detach().item():8.4f} | {infos_str} | t: {timer():8.4f}')

                metrics = {k: v.detach().item() for k, v in infos.items()}
                metrics['train/it'] = self.step
                metrics['train/loss'] = loss.detach().item()
                metrics['train/lr'] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)

            self.step += 1

            # very rough epoch tracking for cycled loader
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
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, epoch):
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath, map_location=self.device)

        self.step = data['step']
        self.unwrap_model(self.model).load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def load4resume(self, loadpath):
        data = torch.load(loadpath, map_location=self.device)
        self.unwrap_model(self.model).load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def get_sample_savedir(self, i):
        div_freq = 50000
        subdir = str((i // div_freq) * div_freq)
        sample_savedir = os.path.join(self.logdir, subdir)
        if not os.path.isdir(sample_savedir):
            os.makedirs(sample_savedir)
        return sample_savedir