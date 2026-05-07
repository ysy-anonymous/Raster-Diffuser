# Adapted from pb diffusion github
import os
import copy
import numpy as np
import torch
import einops

from core.pb_diffusion.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from core.pb_diffusion.utils.timer import Timer
from core.pb_diffusion.utils.train_utils import cycle, EMA
from core.pb_diffusion.utils.train_utils import get_lr
import core.pb_diffusion.networks.diffusion as dmodels
from core.pb_diffusion.utils.train_utils import CosineAnnealingWarmupRestarts
from tqdm import tqdm
import wandb

class TrainerPBDiff(object):
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
        n_samples=2, ## sample times per traj
        num_render_samples=2, ## 
        clip_grad=False,
        clip_grad_max=False,
        n_train_steps=None,
        trainer_dict={},
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
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

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=8, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        
        self.logdir = results_folder
        self.bucket = results_folder
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.num_render_samples = num_render_samples
        self.clip_grad = clip_grad # grad norm 
        self.clip_grad_max = clip_grad_max # max grad
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

        self.reset_parameters()
        self.step = 0
        self.debug_mode = False

        self.device = train_device


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
        
    def prepare_inputs(self, batch):
        action = batch['sample'].to(self.device, dtype=torch.float32)
        map_cond= batch['map'].to(self.device, dtype=torch.float32)
        env_cond = batch['env'].to(self.device, dtype=torch.float32)
                
        return action, map_cond, env_cond

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        wandb.init()

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)

                # batch = batch_to_device(batch, device=self.device)
                batch = self.prepare_inputs(batch)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                
                loss.backward()

                ## gradient clipping
                if self.clip_grad_max:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_max)
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)


            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_warmupDecay:
                self.scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            ## checkdesign
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                # a0 loss is from self.model
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])

                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

                metrics = {k:v.detach().item() for k, v in infos.items()}
                
                metrics['train/it'] = self.step
                metrics['train/loss'] = loss.detach().item()
                metrics['train/lr'] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }

        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        # loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
    
    def load4resume(self, loadpath):
        data = torch.load(loadpath)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def get_sample_savedir(self, i):
        div_freq = 50000
        subdir = str( (i // div_freq) * div_freq )
        sample_savedir = os.path.join(self.logdir, subdir)
        if not os.path.isdir(sample_savedir):
            os.makedirs(sample_savedir)
        return sample_savedir



def vis_preproc_dyn_wtraj(batch, n_samples):
    '''pad wtrajs for dynamic env'''
    wloc = einops.repeat(
                    batch.wall_locations, 'b h d -> (repeat b) h d',repeat=n_samples)

    wloc = np.concatenate([
            wloc[:, 0:1],
            wloc,
            wloc[:, -1:]
        ], axis=1)

    return wloc