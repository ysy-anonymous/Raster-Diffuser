import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

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


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        map_cdim,
        obs_cdim,
        time_cdim,
        map_size,
        patch_size,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = time_cdim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        global_cond_dim = map_cdim + obs_cdim + time_dim
        mish=True

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                FiLMTemporalBlock(dim_in, dim_out, dim_out, gcond_dim=global_cond_dim, kernel_size=5, n_groups=8, mish=mish),
                FiLMTemporalBlock(dim_out, dim_out * 2, dim_out, gcond_dim=global_cond_dim, kernel_size=5, n_groups=8, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        # Attention Layer on UNet latent dimension
        self.patchfier = Patchfier(input_size=map_size, patch_size=patch_size, style=PatchifyStyle.VIT_STYLE.value)
        self.map_k = nn.Linear(in_features=map_cdim * patch_size[0] * patch_size[1], out_features=mid_dim, bias=True)
        self.map_v = nn.Linear(in_features=map_cdim * patch_size[0] * patch_size[1], out_features=mid_dim, bias=True)
        
        self.mid_block1 = FiLMTemporalBlock(mid_dim, mid_dim * 2, mid_dim, gcond_dim=global_cond_dim, kernel_size=5, n_groups=8, mish=mish)
        self.mid_block2 = FiLMTemporalBlock(mid_dim, mid_dim * 2, mid_dim, gcond_dim=global_cond_dim, kernel_size=5, n_groups=8, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                FiLMTemporalBlock(dim_out * 2, dim_out, dim_in, gcond_dim=global_cond_dim, kernel_size=5, n_groups=8, mish=mish),
                FiLMTemporalBlock(dim_in, dim_in * 2, dim_in, gcond_dim=global_cond_dim, kernel_size=5, n_groups=8, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, map_cond, obs_cond):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        spatial, pooled = map_cond
        obs_cond = obs_cond
        h = []
        
        global_cond = torch.cat([t, pooled, obs_cond], dim=1) # [B, map_dim + obs_dim + time_dim]

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, global_cond)
            x = resnet2(x, global_cond)
            h.append(x)
            x = downsample(x)

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

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
