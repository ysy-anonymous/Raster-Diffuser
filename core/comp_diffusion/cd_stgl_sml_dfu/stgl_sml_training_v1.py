import os
import copy
import numpy as np
import torch
import einops, wandb
import pdb

from core.comp_diffusion.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from core.comp_diffusion.helpers import apply_conditioning
from core.comp_diffusion.utils.timer import Timer
from core.comp_diffusion.utils.train_utils import get_lr
from core.comp_diffusion.utils.training import cycle, EMA
from core.comp_diffusion.cd_stgl_sml_dfu import Stgl_Sml_GauDiffusion_InvDyn_V1
import core.comp_diffusion.utils as utils

from diffusers.optimization import get_scheduler
from core.comp_diffusion.datasets.rrt_map.rrt_map_dataset import rrt_comp_collate_fn

class Stgl_Sml_Trainer_v1(object):
    def __init__(
        self,
        diffusion_model: Stgl_Sml_GauDiffusion_InvDyn_V1,
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
        horizon=32, # number of points that consists trajectory
        n_reference=8,
        n_samples=2,
        device='cuda',
        results_folder='./results',
        n_train_steps= 30000, # placeholder for configuration
        trainer_dict={},
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.horizon = horizon

        self.trainer_dict = trainer_dict

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=6, shuffle=True, pin_memory=True, collate_fn=rrt_comp_collate_fn
        )
        print("number of iterations per epoch for constructed dataloader is : ", len(dataloader))
        self.dataloader = cycle(dataloader)
        
        
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder

        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0
        self.device = device

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

        return action, map_cond

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        self.lr_scheduler = get_scheduler(
            name=self.trainer_dict["lr_scheduler"]["name"],
            optimizer=self.optimizer,
            num_warmup_steps=self.trainer_dict["lr_scheduler"]["num_warmup_steps"],
            num_training_steps=n_train_steps,
            num_cycles=self.trainer_dict['lr_scheduler']['num_cycles']
        )
        wandb.init()

        timer = Timer()
        for i_tr in range(n_train_steps):
            for i_ac in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                obs_trajs, map_cond = self.prepare_inputs(batch)
                
                if self.model.tr_cond_type == 'no':
                    cond_st_gl = {}
                else:
                    # construct dictionary for start, goal point location
                    cond_st_gl = {0: obs_trajs[:, 0, :].squeeze(), self.horizon-1: obs_trajs[:, self.horizon-1, :].squeeze()}
                
                # loss, infos = self.model.loss(*batch)
                loss, infos = self.model.loss(x_clean=obs_trajs, cond_st_gl=cond_st_gl, map_cond=map_cond)

                # pdb.set_trace()

                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            # Update LR Scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

                # pdb.set_trace()
                ## save to online
                metrics = {k:v.detach().item() for k, v in infos.items()}
                
                metrics['train/it'] = self.step
                metrics['train/loss'] = loss.detach().item()
                metrics['train/lr'] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        utils.print_color(f'[ utils/training ] Saved model to {savepath}', c='y')
        

    def load4resume(self, loadpath):
        ## Dec 26
        data = torch.load(loadpath)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.step = data['step']
    

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])