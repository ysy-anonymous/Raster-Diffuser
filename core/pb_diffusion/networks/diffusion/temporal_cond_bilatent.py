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
    PositionalEncoding2D,
    Conv1dBlock_dd,
)
import numpy as np
from core.pb_diffusion.networks.diffusion.temporal_dd import ResidualTemporalBlock_dd

# BiLatent Fusion Block
from core.diffuser.networks.diffusion_decoder.helpers.BiFusion import BiDirectionalFusionBlock

# Instead of Wall Location Embedding, Our Tasks use Map and Start/Goal Location as Conditionals
class ResidualTemporalBlock_WCond(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, cond_dim, kernel_size=5, **kwargs):
        """
        embed_dim: input dimension for time_mlp (dimension of time positional embedding)
        wall_embed_dim:
        """
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, global_cond):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            global_cond: [batch_size x (map_dim + obs_dim)]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''

        # Encode Conditionals
        out = self.blocks[0](x) + self.time_mlp(t) + self.cond_mlp(global_cond)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

# --------------------------------------------------------
class TemporalUnet_WCond_BIL(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        map_dim,
        obs_dim,
        bilatent_dim,
        num_iters,
        upsample_hw,
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
        
        if self.cat_t_w:
            time_dim = dim + map_dim + obs_dim
        else:
            time_dim = dim

        ## set param used in ebm
        self.energy_mode = network_config.get('energy_mode', False)
        if self.energy_mode:
            mish = False
            act_fn = nn.SiLU()

            self.energy_param_type = network_config['energy_param_type'] ## should use this line
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
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.network_config = network_config

        ## default no dropout
        self.concept_drop_prob = network_config.get('concept_drop_prob', -1.0)
        self.last_conv_ksize = network_config.get('last_conv_ksize', 1) # 1 is more stable than 5
        self.force_residual_conv = network_config.get('force_residual_conv', False)
        self.time_mlp_config = network_config.get('time_mlp_config', False)
        resblock_config = dict(force_residual_conv=self.force_residual_conv,
                               time_mlp_config=self.time_mlp_config)
        
        assert not self.force_residual_conv, 'must be False'
        assert self.last_conv_ksize == 1, '1 is from diffuser'


        print(f'[TemporalUnet_WCond] concept_drop_prob: {self.concept_drop_prob}')
        print(f'[TemporalUnet_WCond] time_dim: {time_dim}')

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        res_block_type = ResidualTemporalBlock_dd if self.cat_t_w else ResidualTemporalBlock_WCond

        

        self.down_times = network_config.get('down_times', 1e5)
        utils.print_color(f'[Unet down_times] {self.down_times}', c='c')
        ## default in_out: [(64,128), (128,256), (256,512)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind >= self.down_times

            self.downs.append(nn.ModuleList([
                res_block_type(dim_in, dim_out, embed_dim=time_dim, horizon=horizon, cond_dim=map_dim+obs_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize), # ks should be 5 by default
                res_block_type(dim_out, dim_out, embed_dim=time_dim, horizon=horizon, cond_dim=map_dim+obs_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                Downsample1d(dim_out) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, cond_dim=map_dim+obs_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize)
        self.mid_block2 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, cond_dim=map_dim+obs_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind < ( num_resolutions - self.down_times - 1)

            ##? Eg. dim_out:4, dim_in:8, dim_out*2 because we concat residual 
            self.ups.append(nn.ModuleList([
                res_block_type(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon, cond_dim=map_dim+obs_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                res_block_type(dim_in, dim_in, embed_dim=time_dim, horizon=horizon, cond_dim=map_dim+obs_dim, mish=mish, conv_zero_init=self.conv_zero_init, resblock_config=resblock_config, kernel_size=self.resblock_ksize),
                Upsample1d(dim_in) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        ## -- BiLatent Fusion Block --
        self.bilatent_block = BiDirectionalFusionBlock(cm=map_dim, cp=3, cs=bilatent_dim, ct=time_dim, 
                                                           Hs=upsample_hw[0], Ws=upsample_hw[1], R=num_iters, sigma=1.2)
        
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
        
    def set_schedule_parameters(self, betas, alphas, alphas_cumprod):
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

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

        noisy_traj = x
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

        if self.cat_t_w:
            t = torch.cat([t,global_cond], dim=-1)
        
        h = []

        for resnet, resnet2, downsample in self.downs:

            x = resnet(x, t, global_cond)
            x = resnet2(x, t, global_cond)
            h.append(x)
            x = downsample(x)

        # print(f'after downs: {x.shape}')

        x = self.mid_block1(x, t, global_cond)
        x = self.mid_block2(x, t, global_cond)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, global_cond)
            x = resnet2(x, t, global_cond)
            x = upsample(x)

        # print(f'after ups: {x.shape}')

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        
        if self.energy_mode and self.training:
            x = x
        else: # avoid using bilatent fusion when energy_mode=True
            # ====== BiLatent Adjust =======
            alpha_bar = self.alphas_cumprod[time].view(-1, 1, 1) # (B, 1, 1)
            sqrt_ab = alpha_bar.sqrt()
            sqrt_omab = (1.0 - alpha_bar).sqrt()
            eps_pred = x

            x0_hat = (noisy_traj - sqrt_omab * eps_pred) / (sqrt_ab + 1e-8)
            x0_hat = x0_hat.clamp(-1, 1)

            x0_hat_ref = self.bilatent_block(spatial, x0_hat, t)
            x0_hat_ref = x0_hat_ref.clamp(-1, 1)

            # Convert refined x0 -> epsilon
            x = (noisy_traj - sqrt_ab * x0_hat_ref) / (sqrt_omab + 1e-8)

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
