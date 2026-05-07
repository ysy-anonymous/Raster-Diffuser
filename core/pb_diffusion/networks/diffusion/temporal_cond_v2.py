# Adapted from potential-based diffusion 
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
import core.pb_diffusion.utils as utils

from core.pb_diffusion.networks.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Conv1dBlock_dd,
)
import numpy as np
from core.diffuser.networks.utils.helpers import Patchfier
from core.diffuser.networks.utils.enums import PatchifyStyle

# "FiLM" Layer for conditioning
class FiLMTemporalBlock(nn.Module):
    def __init__(self, inp_channels, hid_channels, out_channels, gcond_dim, kernel_size=5, n_groups=8, mish=False, **kwargs):
        """
            inp_channels: Input Channels,
            hid_channels: Hidden Channels between Input and Ouput channels
            out_channels: Output Channels,
            gcond_dim [B, D]: dimension of global condition (concat of map, observation, time
            kernel_size: Conv 1d kernel size
        """
        super().__init__()
        self.inp_channels = inp_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.gcond_dim = gcond_dim
        self.kernel_size = kernel_size
        
        if mish:
            act_fn = nn.Mish
        else:
            act_fn = nn.SiLU
        
        if self.inp_channels != self.out_channels:
            self.res_conv1d = nn.Conv1d(in_channels=inp_channels, out_channels=out_channels, 
                                        kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1d = nn.Conv1d(in_channels=inp_channels, out_channels=hid_channels,
                                kernel_size=kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=False)
        self.gnorm = nn.GroupNorm(num_groups=n_groups, num_channels=hid_channels)
        self.act = act_fn()
        
        self.conv1d_2 = nn.Conv1d(in_channels=hid_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=False)
        self.gnorm_2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
        self.act_2 = act_fn()
        
        self.cond_encoder = nn.Sequential(
            nn.Linear(gcond_dim, gcond_dim),
            act_fn(),
            nn.Linear(gcond_dim, out_channels * 2),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1))
        )
    
    def forward(self, x, cond):
        input = x
        film_feat = self.cond_encoder(cond) # (B, 2 * C, 1)
        film_scale, film_shift = film_feat.chunk(2, dim=1) # (B, C, 1)
        film_scale = film_scale + 1.0 # initialized near identical
        
        x = self.conv1d(x)
        x = self.act(x)
        x = self.gnorm(x)
        
        x = self.conv1d_2(x)
        x = x * film_scale + film_shift # film
        x = self.act_2(x)
        x = self.gnorm_2(x)
        if self.inp_channels == self.out_channels:
            return x + input
        else:
            return x + self.res_conv1d(input)
        
# --------------------------------------------------------
class FiLMTemporalUnet_WAttn(nn.Module):

    def __init__(
        self,
        map_size,
        patch_size,
        horizon,
        transition_dim,
        time_dim,
        map_dim,
        obs_dim,
        dim=32, # may use 64
        dim_mults=(1, 2, 4, 8),
        network_config={},
    ):
        """
        NOTE energy_mode might be implemented only when cat_t_w=True,
        e.g. some zero init is not implied in residual block.
        """
        super().__init__()

        ## dim=64 [2,64*1,64*4,64*8]
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        ## [(64,128), (128,256), (256,512)]
        in_out = list(zip(dims[:-1], dims[1:]))
        utils.print_color(f'[ models/temporal_cond ] Channel dimensions: {in_out}', c='c')

        ## --------- init MLP for time / wall ---------
        ## cat the vector embedding of time and wall before feeding to the MLP
        self.cat_t_w = network_config.get('cat_t_w', False)
        self.resblock_ksize = network_config.get('resblock_ksize', 5) # kernel size for residual block
        self.use_downup_sample = network_config.get('use_downup_sample', True)

        assert self.use_downup_sample and self.resblock_ksize == 5, 'the default settings'
        
        ## set param used in ebm
        self.energy_mode = network_config.get('energy_mode', False)
        if self.energy_mode:
            mish = False
            act_fn = nn.SiLU()

            # Energy_Param_Type must contain 'L2'
            self.energy_param_type = network_config['energy_param_type']
            if 'L2' in self.energy_param_type:
                self.conv_zero_init = network_config.get('conv_zero_init', False)
            else: 
                raise NotImplementedError()
            
            print(f'[ models/temporal_cond ] conv_zero_init {self.energy_param_type} {self.conv_zero_init}')
        else:
            mish = True
            act_fn = nn.Mish()
            self.conv_zero_init = False


        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            act_fn,
            nn.Linear(time_dim * 4, time_dim),
        )

        self.network_config = network_config

        ## default no dropout
        self.concept_drop_prob = network_config.get('concept_drop_prob', -1.0)
        self.last_conv_ksize = network_config.get('last_conv_ksize', 1) # 1 is more stable than 5
        self.force_residual_conv = network_config.get('force_residual_conv', False)
        self.time_mlp_config = network_config.get('time_mlp_config', False)
        
        assert not self.force_residual_conv, 'must be False'
        assert self.last_conv_ksize == 1, '1 is from diffuser'


        print(f'[TemporalUnet_WCond] concept_drop_prob: {self.concept_drop_prob}')
        print(f'[TemporalUnet_WCond] time_dim: {time_dim}')
        
        # UNet Layer Configurations
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        res_block_type = FiLMTemporalBlock
        global_cond_d = map_dim + obs_dim + time_dim

        self.down_times = network_config.get('down_times', 1e5)
        utils.print_color(f'[Unet down_times] {self.down_times}', c='c')
        ## default in_out: [(64,128), (128,256), (256,512)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind >= self.down_times
            
            self.downs.append(nn.ModuleList([
                res_block_type(dim_in, dim_out, dim_out, gcond_dim=global_cond_d, kernel_size=self.resblock_ksize, n_groups=8, mish=mish),
                res_block_type(dim_out, dim_out * 2, dim_out, gcond_dim=global_cond_d, kernel_size=self.resblock_ksize, n_groups=8, mish=mish),
                Downsample1d(dim_out) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        
        # Attention Layer on UNet latent dimension
        self.patchfier = Patchfier(input_size=map_size, patch_size=patch_size, style=PatchifyStyle.VIT_STYLE.value)
        self.map_k = nn.Linear(in_features=map_dim * patch_size[0] * patch_size[1], out_features=mid_dim, bias=True)
        self.map_v = nn.Linear(in_features=map_dim * patch_size[0] * patch_size[1], out_features=mid_dim, bias=True)

        self.mid_block1 = res_block_type(mid_dim, mid_dim * 2, mid_dim, gcond_dim=global_cond_d, kernel_size=self.resblock_ksize, n_groups=8, mish=mish)
        self.mid_block2 = res_block_type(mid_dim, mid_dim * 2, mid_dim, gcond_dim=global_cond_d, kernel_size=self.resblock_ksize, n_groups=8, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind < ( num_resolutions - self.down_times - 1)

            ##? Eg. dim_out:4, dim_in:8, dim_out*2 because we concat residual 
            self.ups.append(nn.ModuleList([
                res_block_type(dim_out * 2, dim_out, dim_in, gcond_dim=global_cond_d, kernel_size=self.resblock_ksize, n_groups=8, mish=mish),
                res_block_type(dim_in, dim_in * 2, dim_in, gcond_dim=global_cond_d, kernel_size=self.resblock_ksize, n_groups=8, mish=mish),
                Upsample1d(dim_in) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2
        
        ## -- Ordinary Diffusion Setup --
        if not self.energy_mode:
            self.final_conv = nn.Sequential(
                Conv1dBlock(dim, dim, kernel_size=self.resblock_ksize), # 5
                nn.Conv1d(dim, transition_dim, 1),
            )
        ## -- Energy Diffusion Parameterization Setup --
        elif self.energy_param_type == 'L2':
            self.final_conv = nn.Sequential(
                Conv1dBlock_dd(dim, dim, kernel_size=5, mish=mish, conv_zero_init=False),
                nn.Conv1d(dim, transition_dim, 1),
            )
        else:
            raise NotImplementedError()


    def forward(self, x, time, map_cond, obs_cond, use_dropout=True,
                force_dropout=False, half_fd=False,):
        '''
            x : [ batch x horizon x transition ]
            time: [batch,]
            map_cond: ([B, C, H, W], [B, dim])
            obs_cond: [B, dim]
            half_fd: drop the conditions for the second half in the input batch 
        '''
        if self.energy_mode:
            x.requires_grad_(True)
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        # Conditionals: Time, Map, Observation
        t = self.time_mlp(time)
        spatial, pooled = map_cond
        obs_cond = obs_cond
        
        global_cond = torch.cat([pooled, obs_cond], dim=1) # [B, map_dim + obs_dim]

        ## drop concept only when training, rand uniform [0, 1)
        if use_dropout:
            assert self.training
            b = global_cond.shape[0]
            global_cond[np.random.rand(b,) < self.concept_drop_prob] = 0.


        if force_dropout:
            assert not self.training
            if half_fd:
                # drop the second half
                assert len(global_cond) % 2 == 0
                global_cond[int(len(global_cond)//2):] = 0. * global_cond[int(len(global_cond)//2):] 
            else:
                global_cond = 0. * global_cond
        global_cond = torch.cat([t,global_cond], dim=-1)
        
        h = []
        for resnet, resnet2, downsample in self.downs:

            x = resnet(x, global_cond)
            x = resnet2(x, global_cond)
            h.append(x)
            x = downsample(x)

        # print(f'after downs: {x.shape}')

        x = self.mid_block1(x, global_cond)
        
        spatial_token = self.patchfier(spatial)
        spa_k = self.map_k(spatial_token)
        spa_v = self.map_v(spatial_token)
        x = x.permute(0, 2, 1) # [B, H, C]
        x = torch.nn.functional.scaled_dot_product_attention(x, spa_k, spa_v) + x
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.mid_block2(x, global_cond)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_cond)
            x = resnet2(x, global_cond)
            x = upsample(x)

        # print(f'after ups: {x.shape}')

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        ## energy_mode will return inside
        if self.energy_mode:
            unet_out = x # B, horizon, dim
            if self.energy_param_type in ['L2',]:
                energy_norm = 0.5 * unet_out * unet_out # should not have neg sign
                energy_norm = energy_norm.sum(dim=(1,2))
            else: 
                raise NotImplementedError()

            
            if not self.training:
                eps = torch.autograd.grad([energy_norm.sum()],[x_inp],create_graph=False)[0]
                # print('energy_norm.sum()', energy_norm.sum())
            else:
                engy_batch = energy_norm.sum()
                eps = torch.autograd.grad([engy_batch,],[x_inp],create_graph=True)[0]
                return eps, engy_batch.detach()

                 
            return eps

        ## final output: B H dim
        # print(f'final output: {x.shape}')

        return x
