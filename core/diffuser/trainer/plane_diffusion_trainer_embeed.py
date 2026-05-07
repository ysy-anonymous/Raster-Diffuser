from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Dict, Optional

from utils.load_utils import build_noise_scheduler_from_config
from core.diffuser.datasets.plane_dataset_embeed import PlanePlanningDataSets
from core.diffuser.networks.diffusion_decoder.helpers.Coarse2FineResHead import interpolate_time


def build_dataloader_from_dataset_and_config(config: Dict, dataset: torch.utils.data.Dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )


class PlaneDiffusionTrainer:
    def __init__(
        self, net: nn.Module, dataset: PlanePlanningDataSets, config: Dict, device: Optional[str] = None
    ):
        self.net = net
        self.config = config.to_dict()
        self.noise_scheduler = build_noise_scheduler_from_config(self.config)
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() and device is None else device

        # if the network has 'set_noise_scheduler' function, call it
        method = getattr(self.net, 'set_noise_scheduler', None)
        if callable(method):
            method(self.noise_scheduler)
        
        # if the network has 'use_aux_loss' variable, read it    
        self.use_aux = getattr(self.net, 'use_aux_loss', None)
        self.T_coarse = getattr(self.net, 'T_coarse', None)
        self.T_mid = getattr(self.net, 'T_mid', None)
        self.T_fine = getattr(self.net, 'T_fine', None)
        
        # move network to device (call this after setting noise scheduler if needed)
        self.net.to(self.device)

        # build optimizer
        if self.config["trainer"]["optimizer"]["name"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=self.net.parameters(),
                lr=self.config["trainer"]["optimizer"]["learning_rate"],
                weight_decay=self.config["trainer"]["optimizer"]["weight_decay"],
            )
        else:
            raise NotImplementedError

        # build dataset
        self.dataloader = build_dataloader_from_dataset_and_config(self.config, dataset)

        # set EMA
        self.use_ema = self.config["trainer"]["use_ema"]
        self.ema = EMAModel(parameters=self.net.parameters(), power=0.75) if self.use_ema else None

    def prepare_inputs(self, batch):
        """
        - sample: noisy sample (B, T, action_dim)
        - map: map information (B, 1, H, W)
        - env: observation [start, goal] -> (B, 2 * obs_dim)
        """
        action = batch["sample"].to(self.device, dtype=torch.float32)     # noisy trajectory (B, T, action_dim) - target action trajectory
        map_cond = batch["map"].to(self.device, dtype=torch.float32)      # condition: map (B, 1, 8, 8) - binary map for 8 x 8 grid
        env_cond = batch["env"].to(self.device, dtype=torch.float32)      # condition: env (start, goal) - start, goal 2D coordinates (B, 2 * obs_dim)
        
        batch_size = action.shape[0]

        return map_cond, env_cond, action, batch_size

    def optimization_step(self, action, map_cond, env_cond, batch_size):
        # sample noise to add to actions
        noise = torch.randn(action.shape, device=self.device) # shape of action: (B, T, 2)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

        # predict the noise residual
        # network predicts the same shape of noise tensor.
        net_out = self.net(noisy_actions, timesteps, map_cond, env_cond)
        if isinstance(net_out, tuple):
            noise_pred = net_out[0] # if model outputs tuple of (noise, trajectory)
        else:
            noise_pred = net_out # if model just outputs noise in pytorch tensor format

        # L2 loss
        if len(noise_pred.shape) == 3: # general noise prediction -> [B, T, 2]
            loss = nn.functional.mse_loss(noise_pred, noise)
        elif len(noise_pred.shape) == 4: # multi-hypothesis noise prediction -> [B, K, T, 2]
            err = ((noise_pred - noise[:, None]) ** 2).mean(dim=(2, 3)) # (B, K)
            loss = err.min(dim=1).values.mean()
            
            # loss_denoise = err.min(dim=1).values.mean()
            # k_star = err.argmin(dim=1) # (B,)
            # loss_sel = F.cross_entropy(logits, k_star)
            # loss = loss_denoise + loss_sel * self.lambda_
            
        if self.use_aux:
            if self.T_coarse and self.T_mid: # Multi Resolution timestep based loss
                aux = net_out[2]
                x0_coarse = interpolate_time(action, self.T_coarse)
                x0_mid = interpolate_time(action, self.T_mid)
                loss_aux_1 = F.mse_loss(aux['traj_coarse'], x0_coarse)
                loss_aux_2 = F.mse_loss(aux['traj_mid'], x0_mid)
                loss_aux_3 = F.mse_loss(aux['traj_fine'], action)
                loss = loss + loss_aux_1 * 0.1 + loss_aux_2 * 0.1 + loss_aux_3
                
                
        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        self.lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        if self.use_ema:
            self.ema.step(self.net.parameters())

        return loss

    def train(self, num_epochs: int, save_ckpt_epoch: int=None, save_path: str ="/exhdd/seungyu/diffusion_motion/trained_weights"):
        if save_ckpt_epoch is None:
            save_ckpt_epoch = num_epochs

        # set learning rate scheduler
        self.lr_scheduler = get_scheduler(
            name=self.config["trainer"]["lr_scheduler"]["name"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["trainer"]["lr_scheduler"]["num_warmup_steps"],
            num_training_steps=len(self.dataloader) * num_epochs,
            num_cycles=self.config['trainer']['lr_scheduler']['num_cycles']
        )

        # training loop
        trn_loss = []
        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(self.dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        map_cond, env_cond, action, B = self.prepare_inputs(nbatch)
                        loss = self.optimization_step(action, map_cond, env_cond, B)

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                trn_loss.append(np.mean(epoch_loss))

                # save intermediate ckpt
                if (epoch_idx + 1) % save_ckpt_epoch == 0:
                    self.save_checkpoint(path=f"{save_path}/ckpt_ep{epoch_idx}.ckpt")

        return trn_loss

    def save_checkpoint(self, path: str):
        save_model = self.net
        if self.config["trainer"]["use_ema"]:
            self.ema.copy_to(save_model.parameters())
        torch.save(save_model.state_dict(), path)