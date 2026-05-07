import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from enum import Enum

from core.diffuser.networks.helpers import SinusoidalPosEmb
from core.diffuser.networks.diffusion_decoder.helpers.MultiHypothesisRefiner import MultiHypothesisRefiner
from core.diffuser.networks.utils.helpers import ChannelFirstLayerNorm

from core.diffuser.networks.diffusion_decoder.building_block.convblocks import Conv1DResBlock, Conv1DBlock, DownSampler, UpSampler
from core.diffuser.networks.utils.helpers import Patchfier
from core.diffuser.networks.utils.enums import PatchifyStyle


# map features: (B, 512, 32, 32)
# env features: (B, 256)
# timestep features: (B, 256)
# input features (noisy trajectorys): (B, T=32, 2)
class MultiHypothesisDiffusionTrajDecoder(nn.Module):
    """
        Conditional (film-based) 1D Unet based decoder.
    """
    def __init__(self, stem_channels: int, channel_scheme: List, kernel_scheme: List, stride_scheme: List,
                 padding_scheme: List, out_padding_scheme: List, group_scheme: List, activation, norm_layer, n_groups: int, 
                 timestep_dim:int, map_cond_dim:int, obs_dim: int, map_size:tuple, patch_size: tuple,
                 num_hypothesis: int, token_dim: int, hidden_dim:int, num_iters: int):
        super().__init__()
        self.activation = activation  
        cond_dim = map_cond_dim + timestep_dim + obs_dim # global condition dimension
        
        self.pooled_traj_stem = nn.ModuleList([
            Conv1DBlock(in_channels=map_cond_dim, out_channels=map_cond_dim, kernel_size=5, stride=1,
                        padding='same', dilation=1, groups=1, activation=activation, norm_layer=norm_layer, n_groups=n_groups),
            Conv1DResBlock(in_channels=map_cond_dim, hidden_channels=map_cond_dim * 2, out_channels=map_cond_dim, kernel_size=5,
                           stride=1, groups=1, padding='same', activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups),
            Conv1DResBlock(in_channels=map_cond_dim, hidden_channels=map_cond_dim * 2, out_channels=map_cond_dim, kernel_size=5,
                           stride=1, groups=1, padding='same', activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups),
            Conv1DResBlock(in_channels=map_cond_dim, hidden_channels=map_cond_dim * 2, out_channels=map_cond_dim, kernel_size=5,
                           stride=1, groups=1, padding='same', activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups),
            nn.Conv1d(in_channels=map_cond_dim, out_channels=stem_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        ])
        
        # diffusion time-step encoder
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_dim),
            nn.Linear(timestep_dim, timestep_dim * 4),
            activation(),
            nn.Linear(timestep_dim * 4, timestep_dim),
        )
        
        # stem layers for trajectory decoding
        self.traj_stem = nn.Sequential(
            Conv1DBlock(in_channels=2, out_channels=32, kernel_size=5,
                        stride=1, padding='same', dilation=1, groups=1, activation=activation, norm_layer=norm_layer, n_groups=n_groups),
            Conv1DBlock(in_channels=32, out_channels=stem_channels, kernel_size=5,
                        stride=1, padding='same', dilation=1, groups=1, activation=activation, norm_layer=norm_layer, n_groups=n_groups)
        )
        
        
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        depth = len(channel_scheme)
        all_channel = [stem_channels] + channel_scheme
        for i in range(depth):
            self.encoder.append(DownSampler(in_channels=all_channel[i], out_channels=all_channel[i+1], kernel_size=kernel_scheme[i], 
                                            stride=stride_scheme[i], padding=padding_scheme[i], groups=group_scheme[i],
                                            activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups))
        
        last_channel = all_channel[-1]
        for i, (ker_sz, stride, padding, out_padding, group) in enumerate(reversed(list(zip(kernel_scheme, stride_scheme, padding_scheme, out_padding_scheme, group_scheme)))):
            self.decoder.append(UpSampler(in_channels=last_channel, out_channels=all_channel[-2-i], kernel_size=ker_sz, 
                                          stride=stride, padding=padding, out_padding=out_padding, groups=group,
                                          activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups))
            last_channel = all_channel[-2-i] * 2
        
        # multi-hypothesis (K) based refiner
        # this requires best K-min loss adaptation instead of simple mse/mae loss.
        self.dim_adjuster = nn.Linear(in_features=last_channel, out_features=hidden_dim, bias=True)
        self.multi_hypo_refiner = MultiHypothesisRefiner(K=num_hypothesis, token_dim=token_dim, cm=map_cond_dim,
                                                    ch=hidden_dim, dt=timestep_dim, do=obs_dim, R=num_iters)
        
        self.map_patchfier = Patchfier(input_size=map_size, patch_size=patch_size, style=PatchifyStyle.VIT_STYLE.value)
        self.map_linear_k = nn.Linear(in_features=map_cond_dim * patch_size[0] * patch_size[1], out_features=channel_scheme[-1], bias=True)
        self.map_linear_v = nn.Linear(in_features=map_cond_dim * patch_size[0] * patch_size[1], out_features=channel_scheme[-1], bias=True)
        
        # self.init_weight()
    
    def set_noise_scheduler(self, noise_scheduler):
        if noise_scheduler is not None:
            self.noise_scheduler = noise_scheduler
        else:
            print("Input noise scheduler is None.")
        
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod.clone(), persistent=False)
        
    # # initialize weights
    # def init_weight(self):
    #     def _init(m: nn.Module):
    #         # ---- Convs / Linears ----
    #         if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.Linear)):
    #             if self.activation in [nn.ReLU, nn.LeakyReLU]:
    #                 nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    #             else:
    #                 nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         # ---- Norm layers ----
    #         elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, ChannelFirstLayerNorm)):
    #             if getattr(m, "weight", None) is not None:
    #                 nn.init.ones_(m.weight)
    #             if getattr(m, "bias", None) is not None:
    #                 nn.init.zeros_(m.bias)
            
    #         # ---- Embeddings (if any appear inside UNet etc.) ----
    #         elif isinstance(m, nn.Embedding):
    #             nn.init.trunc_normal_(m.weight, std=0.02)
        
    #     # Initialize everything inside this modules (stem + all UNets + modulator)
    #     self.apply(_init)
        
    def forward(self, 
                sample,
                timestep,
                map_cond, 
                obs_cond):
        """
            sample: noisy trajectory (sampled from gt trajectory coordinate) (B, 32, 2) - they are normalized to [-1, 1] from input already.
            map_cond: tuple that has shape [(B, C, H, W), (B, C)]
            obs_cond: observation condtionals that has shape (B, D)
        """
        # Map Conditionals
        map_feat, map_pool_feat = map_cond
        
        # initial noisy sample
        x_t_in = sample
        
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
        traj_x = self.pooled_traj_stem[0](traj_sample)
        traj_x = self.pooled_traj_stem[1](traj_x, cond)
        traj_x = self.pooled_traj_stem[2](traj_x, cond)
        traj_x = self.pooled_traj_stem[3](traj_x, cond)
        traj_x = self.pooled_traj_stem[4](traj_x)
        sample = sample + traj_x # add sampled trajectory feature
        
        
        # ===================== Downsample =====================
        downsampled_feats = []
        # downsampled_pooled_feats = []
        for i in range(len(self.encoder)):
            downsampled_feats.append(sample)
            sample = self.encoder[i](sample, cond)
        
        # ===================== Latent =====================
        map_token = self.map_patchfier(map_feat) # (B, 4, 16 * 64)
        map_token_k = self.map_linear_k(map_token) # (B, 4, channel_scheme[-1]) as key, value
        map_token_v = self.map_linear_v(map_token)
        sample = sample.permute(0, 2, 1) # (B, 32, channel_scheme[-1]) as query
        sample = F.scaled_dot_product_attention(sample, map_token_k, map_token_v) + sample # residual connection
        sample = sample.permute(0, 2, 1).contiguous() # (B, channel_scheme[-1], 32)
            
        # ===================== Upsample =====================
        # Unet Decoder
        last_idx = len(downsampled_feats)-1
        for i in range(len(self.decoder)):
            sample = self.decoder[i](sample, cond)
            sample = torch.cat([sample, downsampled_feats[last_idx-i]], dim=1) # (B, C, L)
            
        sample = sample.permute(0, 2, 1) # (B, C, L) -> (B, L, C)
        input_sample = self.dim_adjuster(sample)
        eps, x0 = self.multi_hypo_refiner(h_base=input_sample, M=map_feat, global_cond=cond, x_t_in=x_t_in, alphas_cumprod=self.alphas_cumprod, timesteps=timesteps)
        # eps, x0, logits = self.multi_hypo_refiner(h_base=input_sample, M=map_feat, global_cond=cond, x_t_in=x_t_in, alphas_cumprod=self.alphas_cumprod, timesteps=timesteps)
        return eps, x0 # (B, K, L, 2) 
        # return eps, x0, logits
    

def main():    
    import torchinfo
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    diffusion_decoder = MultiHypothesisDiffusionTrajDecoder(stem_channels=64, channel_scheme=[128, 256, 512], kernel_scheme=[2, 2, 2], 
                                            stride_scheme=[2, 2, 2], padding_scheme=[0, 0, 0], out_padding_scheme=[0, 0, 0], 
                                            group_scheme=[1, 1, 1], activation=nn.ReLU, norm_layer=ChannelFirstLayerNorm, n_groups=8,
                                            timestep_dim=64, map_cond_dim=128, obs_dim=128, map_size=(8, 8), patch_size=(4, 4), 
                                            num_hypothesis=8, token_dim=256, hidden_dim=512, num_iters=3)
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

    noise, traj = diffusion_decoder(sample=sample, timestep=timestep, map_cond=map_cond, obs_cond=obs_cond)
    print("noise output: ", noise.shape)
    print("trajectory output: ", traj.shape)

if __name__ == '__main__':
    main()