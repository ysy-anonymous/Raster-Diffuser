import os
import copy
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

from diffusers.optimization import get_scheduler
from core.rediffuser.networks.diffuser.utils.timer import Timer
from core.rediffuser.networks.diffuser.utils.cloud import sync_logs


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class ReDiffDDPTrainer(object):
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
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        bucket=None,
        n_train_steps=None,
        train_device=None,
        trainer_dict={},
        use_ddp=False,
        ddp_backend='nccl',
        num_workers=1,
        pin_memory=True,
        find_unused_parameters=False,
        use_amp=True,
        amp_dtype='fp16',   # 'fp16' or 'bf16'
        max_grad_norm=None,
    ):
        super().__init__()

        self.use_ddp = use_ddp
        self.ddp_backend = ddp_backend

        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_main_process = True

        if self.use_ddp:
            if not dist.is_initialized():
                dist.init_process_group(backend=self.ddp_backend)

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.is_main_process = self.rank == 0

            if train_device is None:
                train_device = f"cuda:{self.local_rank}"

        self.device = torch.device(
            train_device if train_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        
        # AMP setup
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.amp_dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16
        # GradScaler is only needed for fp16, not bf16
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.max_grad_norm = max_grad_norm

        self.model = diffusion_model.to(self.device)

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=find_unused_parameters,
            )

        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.sampler = None
        if self.use_ddp:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )

        self.dataloader = cycle(
            DataLoader(
                self.dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                shuffle=(self.sampler is None),
                sampler=self.sampler,
                pin_memory=pin_memory,
            )
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
        
        self.trainer_dict = trainer_dict
        if self.trainer_dict:
            self.lr_scheduler = get_scheduler(
                name=self.trainer_dict["lr_scheduler"]["name"],
                optimizer=self.optimizer,
                num_warmup_steps=self.trainer_dict["lr_scheduler"]["num_warmup_steps"],
                num_training_steps=n_train_steps,
                num_cycles=self.trainer_dict['lr_scheduler']['num_cycles']
            )
        else: self.lr_scheduler=None

        self.logdir = results_folder
        self.bucket = bucket
        self.n_train_steps = n_train_steps

        os.makedirs(self.logdir, exist_ok=True)

        self.reset_parameters()
        self.step = 0

    @property
    def model_module(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model_module.state_dict())

    @torch.no_grad()
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model_module)

    def prepare_inputs(self, batch):
        action = batch['sample'].to(self.device, dtype=torch.float32, non_blocking=True)
        map_cond = batch['map'].to(self.device, dtype=torch.float32, non_blocking=True)
        env_cond = batch['env'].to(self.device, dtype=torch.float32, non_blocking=True)
        return action, map_cond, env_cond

    def reduce_dict(self, input_dict):
        if not self.use_ddp:
            return {
                k: (v.item() if torch.is_tensor(v) else float(v))
                for k, v in input_dict.items()
            }

        with torch.no_grad():
            reduced = {}
            for k, v in input_dict.items():
                if not torch.is_tensor(v):
                    v = torch.tensor(v, device=self.device, dtype=torch.float32)
                else:
                    v = v.detach().to(self.device, dtype=torch.float32)
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                v = v / self.world_size
                reduced[k] = v.item()
        return reduced

    def train(self, n_train_steps):
        timer = Timer()

        for step in range(n_train_steps):
            if self.use_ddp and self.sampler is not None:
                self.sampler.set_epoch(step)

            self.optimizer.zero_grad(set_to_none=True)

            last_infos = None
            last_loss_for_log = None

            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = self.prepare_inputs(batch)

                with autocast(
                    device_type='cuda',
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    loss, infos = self.model_module.loss(*batch)
                    loss = loss / self.gradient_accumulate_every

                self.scaler.scale(loss).backward()

                last_loss_for_log = loss.detach()
                last_infos = infos

            if self.max_grad_norm is not None:
                # unscale before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update LR Scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0 and self.is_main_process:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                reduced_infos = self.reduce_dict(last_infos if last_infos is not None else {})
                reduced_loss = self.reduce_dict({
                    "loss": last_loss_for_log.item() if last_loss_for_log is not None else 0.0
                })["loss"]

                if self.is_main_process:
                    infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in reduced_infos.items()])
                    print(f'{self.step}: {reduced_loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            self.step += 1

    def save(self, epoch):
        data = {
            'step': self.step,
            'model': self.model_module.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch, map_location=None):
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        if map_location is None:
            map_location = self.device

        data = torch.load(loadpath, map_location=map_location)

        self.step = data['step']
        self.model_module.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

        if data.get('scaler', None) is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(data['scaler'])

    def cleanup(self):
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()