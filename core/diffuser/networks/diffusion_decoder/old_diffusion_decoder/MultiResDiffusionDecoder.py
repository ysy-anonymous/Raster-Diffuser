import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from enum import Enum

from core.diffuser.networks.helpers import SinusoidalPosEmb
from core.diffuser.networks.diffusion_decoder.helpers.Coarse2FineResHead import Coarse2FineRefiner
from core.diffuser.networks.utils.helpers import ChannelFirstLayerNorm

from core.diffuser.networks.diffusion_decoder.building_block.convblocks import Conv1DResBlock, Conv1DBlock, DownSampler, UpSampler


# map features: (B, 512, 32, 32)
# env features: (B, 256)
# timestep features: (B, 256)
# input features (noisy trajectorys): (B, T=32, 2)
class MultiResDiffusionDecoder(nn.Module):
    """
        Conditional (film-based) 1D Unet based decoder.
    """
    def __init__(self, stem_channels: int, channel_scheme: List, kernel_scheme: List, stride_scheme: List,
                 padding_scheme: List, out_padding_scheme: List, group_scheme: List, activation, norm_layer, timestep_dim:int, map_cond_dim:int, cond_dim: int,
                 T_coarse: int, T_mid: int, T_fine: int, refiner_in_dim: int, refiner_hidden_dim: int, use_aux_loss: bool):
        super().__init__()
        self.activation = activation
        
        # trajectory feature encoder
        self.pooled_traj_stem = nn.ModuleList([
            Conv1DResBlock(in_channels=map_cond_dim, hidden_channels=map_cond_dim*2, out_channels=map_cond_dim,
                           kernel_size=3, stride=1, groups=1, padding='same', activation=activation, norm_layer=norm_layer, cond_dim=cond_dim),
            Conv1DResBlock(in_channels=map_cond_dim, hidden_channels=map_cond_dim*2, out_channels=stem_channels,
                           kernel_size=3, stride=1, groups=1, padding='same', activation=activation, norm_layer=norm_layer, cond_dim=cond_dim)])
                
        self.shortcut2head = nn.Conv1d(in_channels=stem_channels, out_channels=stem_channels * 2, kernel_size=3,
                                       stride=1, padding=1, dilation=1, groups=1, bias=True)
        
        # diffusion time-step encoder
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_dim),
            nn.Linear(timestep_dim, timestep_dim * 4),
            activation(),
            nn.Linear(timestep_dim * 4, timestep_dim),
        )
        
        # step layers for trajectory decoding
        self.traj_stem = nn.Sequential(
            Conv1DBlock(in_channels=2, out_channels=32, kernel_size=3, # trajectory coordinates are represented with (x, y) points (=2 channels)
                        stride=1, padding='same', dilation=1, groups=1, activation=activation, norm_layer=norm_layer),
            Conv1DBlock(in_channels=32, out_channels=stem_channels, kernel_size=3, 
                        stride=1, padding='same', dilation=1, groups=1, activation=activation, norm_layer=norm_layer)
        )
        
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.pooled_feat_encoder=nn.ModuleList([])
        self.pooled_feat_decoder=nn.ModuleList([])
        depth = len(channel_scheme)
        all_channel = [stem_channels] + channel_scheme
        for i in range(depth):
            self.encoder.append(DownSampler(in_channels=all_channel[i], out_channels=all_channel[i+1], kernel_size=kernel_scheme[i], 
                                            stride=stride_scheme[i], padding=padding_scheme[i], groups=group_scheme[i],
                                            activation=activation, norm_layer=norm_layer, cond_dim=cond_dim))
            self.pooled_feat_encoder.append(DownSampler(in_channels=all_channel[i], out_channels=all_channel[i+1], kernel_size=kernel_scheme[i],
                                                         stride=stride_scheme[i], padding=padding_scheme[i], groups=group_scheme[i],
                                                         activation=activation, norm_layer=norm_layer, cond_dim=cond_dim))
            
        self.latent_process = Conv1DBlock(in_channels=all_channel[-1], out_channels=all_channel[-1], kernel_size=3,
                                          stride=1, padding='same', dilation=1, groups=1, activation=activation, norm_layer=norm_layer)
        self.pooled_feat_latent = Conv1DBlock(in_channels=all_channel[-1], out_channels=all_channel[-1], kernel_size=3,
                                              stride=1, padding='same', dilation=1, groups=1, activation=activation, norm_layer=norm_layer)
        
        last_channel = all_channel[-1]
        for i, (ker_sz, stride, padding, out_padding, group) in enumerate(reversed(list(zip(kernel_scheme, stride_scheme, padding_scheme, out_padding_scheme, group_scheme)))):
            self.decoder.append(UpSampler(in_channels=last_channel, out_channels=all_channel[-2-i], kernel_size=ker_sz, 
                                          stride=stride, padding=padding, out_padding=out_padding, groups=group,
                                          activation=activation, norm_layer=norm_layer, cond_dim=cond_dim))
            self.pooled_feat_decoder.append(UpSampler(in_channels=last_channel, out_channels=all_channel[-2-i], kernel_size=ker_sz,
                                                      stride=stride, padding=padding, out_padding=out_padding, groups=group,
                                                      activation=activation, norm_layer=norm_layer, cond_dim=cond_dim))
            last_channel = all_channel[-2-i] * 2
        
        # multi-time resolution (coarse to fine) trajectory head for sequential trajectory refinement
        self.bridge = nn.Linear(last_channel, refiner_in_dim) # channel adjuster between unet output and refiner input
        self.coarse2fine_head = Coarse2FineRefiner(T_coarse=T_coarse, T_mid=T_mid,
                                                   T_fine=T_fine, in_feat_dim=refiner_in_dim, map_dim=map_cond_dim,
                                                   global_cond_dim=cond_dim, hidden_dim=refiner_hidden_dim)
        
        self.init_weight()

    def set_noise_scheduler(self, noise_scheduler):
        if noise_scheduler is not None:
            self.noise_scheduler = noise_scheduler
        else:
            print("Input noise scheduler is None.")
        
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod.clone(), persistent=False)
        
    # initialize weights
    def init_weight(self):
        def _init(m: nn.Module):
            # ---- Convs / Linears ----
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.Linear)):
                if self.activation in [nn.ReLU, nn.LeakyReLU]:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # ---- Norm layers ----
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, ChannelFirstLayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            
            # ---- Embeddings (if any appear inside UNet etc.) ----
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
        
        # Initialize everything inside this modules (stem + all UNets + modulator)
        self.apply(_init)
        
    def forward(self, 
                sample,
                timestep,
                map_cond, 
                obs_cond):
        """
            sample: noisy trajectory (sampled from gt trajectory coordinate) (B, 32, 2) - they are normalized to [-1, 1] from input already.
            map_cond: map conditionals that has shape (B, C, H, W)
            obs_cond: observation condtionals that has shape (B, D)
        """
        # Map Conditionals
        map_feat, map_pool_feat = map_cond
        
        x_t_in = sample # input noisy trajectory 

        # trajectory feature: (B, 32, 2) -> (B, 32, 1, 2)
        grid_x = sample.unsqueeze(2)
        # trajectory sample: (B, C, 32, 1)
        traj_sample = F.grid_sample(input=map_feat, grid=grid_x, align_corners=False, mode='bilinear', padding_mode='zeros')
        traj_sample = traj_sample.squeeze(-1) # (B, C, 32)
        
        # (B, L, C) -> (B, C, L): convert tensor format to be compatible with conv1d
        sample = sample.permute(0, 2, 1)
        sample = self.traj_stem(sample)
        
        # diffusion timestep encoding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_embed = self.diffusion_step_encoder(timesteps)
        
        cond = torch.cat([map_pool_feat, obs_cond, time_embed], dim=-1) # [B, C + D + D2]
        
        # trajectory encoder
        traj_feat = self.pooled_traj_stem[0](traj_sample, cond) # (B, C, 32)
        traj_x = self.pooled_traj_stem[1](traj_feat, cond) # (B, C, 32)
        sample = sample + traj_x # add sampled trajectory feature
        shortcut = self.shortcut2head(traj_x)
        
        # Unet Decoder
        downsampled_feats = []
        downsampled_pooled_feats = []
        for i in range(len(self.encoder)):
            downsampled_feats.append(sample)
            sample = self.encoder[i](sample, cond)
            downsampled_pooled_feats.append(traj_x)
            traj_x = self.pooled_feat_encoder[i](traj_x, cond)
            sample = sample + traj_x
        
        sample = self.latent_process(sample)
        traj_x = self.pooled_feat_latent(traj_x)
            
        last_idx = len(downsampled_feats)-1
        for i in range(len(self.decoder)):
            sample = self.decoder[i](sample, cond)
            sample = torch.cat([sample, downsampled_feats[last_idx-i]], dim=1) # (B, C, L)
            traj_x = self.pooled_feat_decoder[i](traj_x, cond)
            traj_x = torch.cat([traj_x, downsampled_pooled_feats[last_idx-i]], dim=1)
            sample = sample + traj_x
            
        sample = sample + shortcut # (B, C, T)
        sample = sample.permute(0, 2, 1) # (B, T, C)
        
        # apply coarse-to-fine refiner head
        refiner_feat = self.bridge(sample)
        noise, traj, aux = self.coarse2fine_head(in_feat=refiner_feat, map_cond=map_feat, 
                                                 global_cond=cond, x_t_in=x_t_in, timesteps=timesteps, alphas_cumprod=self.alphas_cumprod)

        return noise, traj, aux
    

def main():    
    import torchinfo
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    diffusion_decoder = MultiResDiffusionDecoder(stem_channels=64, channel_scheme=[128, 256, 512], kernel_scheme=[2, 2, 2], 
                                            stride_scheme=[2, 2, 2], padding_scheme=[0, 0, 0], out_padding_scheme=[0, 0, 0], 
                                            group_scheme=[1, 1, 1], activation=nn.ReLU, norm_layer=ChannelFirstLayerNorm, timestep_dim=64, cond_dim=320,
                                            map_cond_dim=128,T_coarse=8, T_mid=16, T_fine=32, refiner_in_dim=256, refiner_hidden_dim=384, use_aux_loss=False)

    noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon')
    diffusion_decoder.set_noise_scheduler(noise_scheduler)

    diffusion_decoder = diffusion_decoder.to('cuda')
    torchinfo.summary(diffusion_decoder)
    sample = torch.randn(16, 32, 2).to('cuda')
    timestep = torch.tensor([10]).to('cuda')
    map_cond = (torch.randn(16, 128, 32, 32).to('cuda'), torch.randn(16, 128).to('cuda'))
    obs_cond = torch.randn(16, 128).to('cuda')

    noise, traj, aux = diffusion_decoder(sample=sample, timestep=timestep, map_cond=map_cond, obs_cond=obs_cond)
    print("noise out shape: ", noise.shape)
    print("traj out shape: ", traj.shape)
    print("aux: ", aux)

if __name__ == '__main__':
    main()
