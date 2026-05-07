# Adapted From comp_diffuser_release github repo: https://github.com/devinluo27/comp_diffuser_release
# Changes: Add Conditionals for Map, Obs (in minimal way, no FiLM module used)

import numpy as np
import torch, pdb, einops
import torch.nn as nn
from einops.layers.torch import Rearrange

import core.comp_diffusion.utils as utils
from core.comp_diffusion.helpers import (
    SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock,
)

from core.comp_diffusion.cond_cp_dfu.sml_helpers import Traj_Time_Encoder
from core.comp_diffusion.og_models.dit_1d_traj_encoder import DiT1D_Traj_Time_Encoder

from core.diffuser.networks.utils.helpers import Patchfier
from core.diffuser.networks.utils.enums import PatchifyStyle

# --------------------------------------------------------


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
        # film_scale = film_scale + 1.0 # initialized near identical
        
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



class Unet1D_TjTi_Stgl_Cond_V2(nn.Module):

    def __init__(
        self,
        map_size,
        patch_size,
        horizon,
        transition_dim,
        base_dim=32, # may use 64
        dim_mults=(1, 2, 4, 8),
        time_dim=32,
        map_cond_dim=64, # map condition dimension
        network_config={},
    ):
        """
        the UNet Based Denoiser Backbone of CompDiffuser
        use inpainting for the start and goal state;
        also support noisy-sample conditioning.

        """
        super().__init__()

        ## dim=64 [2,64*1,64*4,64*8]
        dims = [transition_dim, *map(lambda m: base_dim * m, dim_mults)]
        ## [(64,128), (128,256), (256,512)]
        in_out = list(zip(dims[:-1], dims[1:]))
        utils.print_color(f'[ models/Unet1D_TjTi_Stgl_Cond_V1 ] Channel dimensions: {in_out}', c='c')

        ## --------- init MLP for time / wall ---------
        ## cat the vector embedding of time and wall before feeding to the MLP
        self.cat_t_w = network_config['cat_t_w'] ## True
        self.resblock_ksize = network_config.get('resblock_ksize', 5) # kernel size for residual block
        self.use_downup_sample = network_config.get('use_downup_sample', True)


        self.st_ovlp_model_config = network_config['st_ovlp_model_config']
        self.end_ovlp_model_config = network_config['end_ovlp_model_config']
        # self.st_ovlp_model_config['in_dim'] = transition_dim
        
        
        self.ext_cond_dim = network_config['ext_cond_dim']
        if network_config.get('ovlp_model_type', 'unet') == 'unet':
            self.st_ovlp_model = Traj_Time_Encoder(**self.st_ovlp_model_config)
            self.end_ovlp_model = Traj_Time_Encoder(**self.end_ovlp_model_config)
        elif network_config['ovlp_model_type'] == 'dit_enc':
            ## Dec 24, DiT-based encoder
            self.st_ovlp_model = DiT1D_Traj_Time_Encoder(**self.st_ovlp_model_config)
            self.end_ovlp_model = DiT1D_Traj_Time_Encoder(**self.end_ovlp_model_config)
            # pdb.set_trace()
        else: 
            raise NotImplementedError
        

        self.network_config = network_config
        ### ------ For inpainting start and goal -------
        self.st_inpaint_model = nn.Identity()
        self.end_inpaint_model = nn.Identity()
        self.inpaint_token_dim = self.network_config['inpaint_token_dim'] ## e.g., 32
        self.inpaint_token_type = self.network_config['inpaint_token_type'] ## e.g., const
        if self.inpaint_token_type == 'const':
            self.st_use_inpaint_token: torch.Tensor
            self.register_buffer('st_use_inpaint_token', \
                                 torch.full(size=(1,self.inpaint_token_dim), fill_value=1., dtype=torch.float32) )

            self.st_no_inpaint_token: torch.Tensor
            self.register_buffer('st_no_inpaint_token', \
                             torch.full(size=(1,self.inpaint_token_dim), fill_value=0., dtype=torch.float32) )

            self.end_use_inpaint_token: torch.Tensor
            self.register_buffer('end_use_inpaint_token', 
                                 torch.full(size=(1,self.inpaint_token_dim), fill_value=1., dtype=torch.float32) )
            
            self.end_no_inpaint_token: torch.Tensor
            self.register_buffer('end_no_inpaint_token', 
                             torch.full(size=(1,self.inpaint_token_dim), fill_value=0., dtype=torch.float32) )

        else:
            raise NotImplementedError
        ### --------------------------------------------

        ##
        wall_embed_dim = self.st_ovlp_model.out_dim + self.end_ovlp_model.out_dim

        assert wall_embed_dim == self.ext_cond_dim ## seems useless below, can be delete?

        assert self.use_downup_sample and self.resblock_ksize == 5, 'the default settings'
        
        if self.cat_t_w:
            # for this task, map conditional, observation conditional is used also.
            tot_cond_dim = time_dim + wall_embed_dim + 2 * self.inpaint_token_dim + map_cond_dim
        else:
            raise NotImplementedError
            time_dim = dim

        # pdb.set_trace() ## check above

        ## set param used in ebm
        self.energy_mode = network_config['energy_mode']
        if self.energy_mode:
            raise NotImplementedError
        else:
            mish = True
            act_fn = nn.Mish()
            self.conv_zero_init = False

        
        ## Luo: just make it deeper
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim),
        )
        # pdb.set_trace() ## check dim


        ## default no dropout
        # self.concept_drop_prob = network_config['concept_drop_prob'] # -1.0
        self.last_conv_ksize = network_config.get('last_conv_ksize', 1) # 1 is more stable than 5
        self.force_residual_conv = network_config.get('force_residual_conv', False)
        self.time_mlp_config = network_config.get('time_mlp_config', False)
        assert self.time_mlp_config == 3
        resblock_config = dict(force_residual_conv=self.force_residual_conv,
                               time_mlp_config=self.time_mlp_config)
        
        assert not self.force_residual_conv, 'must be False'
        assert self.last_conv_ksize == 1, '1 is from diffuser'


        # print(f'[TemporalUnet_WCond] concept_drop_prob: {self.concept_drop_prob}')
        utils.print_color(f'[TemporalUnet_WCond] {time_dim=}, {tot_cond_dim=}, {self.ext_cond_dim=}')
        # pdb.set_trace()
        self.input_t_type = '1d'
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        ## num_resolutions is the number of layer in UNet?
        print('[TemporalUnet_WCond]: in_out: ', in_out,)

        # only ResidualTemporalBlock_dd as convolution choice -> but we need to add conditionals encoders to give the condition information in diffusion process.
        res_block_type = FiLMTemporalBlock

        

        self.down_times = network_config.get('down_times', 1e5)
        utils.print_color(f'[Unet down_times] {self.down_times}', c='c')
        ## default in_out: [(64,128), (128,256), (256,512)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind >= self.down_times

            ## wall_embed_dim seems useless
            self.downs.append(nn.ModuleList([
                # Wall_Embed_dim argument is in **kwargs
                res_block_type(dim_in, dim_out, dim_out, gcond_dim=tot_cond_dim, kernel_size=self.resblock_ksize, n_groups=8, mish=mish), # ks should be 5 by default
                res_block_type(dim_out, dim_out * 2, dim_out, gcond_dim=tot_cond_dim, kernel_size=self.resblock_ksize, n_groups=8, mish=mish),
                Downsample1d(dim_out) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        
        # Attention Layer on UNet latent dimension
        self.patchfier = Patchfier(input_size=map_size, patch_size=patch_size, style=PatchifyStyle.VIT_STYLE.value)
        self.map_k = nn.Linear(in_features=map_cond_dim * patch_size[0] * patch_size[1], out_features=mid_dim, bias=True)
        self.map_v = nn.Linear(in_features=map_cond_dim * patch_size[0] * patch_size[1], out_features=mid_dim, bias=True)
        
        self.mid_block1 = res_block_type(mid_dim, mid_dim * 2, mid_dim, gcond_dim=tot_cond_dim, kernel_size=self.resblock_ksize, n_groups=8, mish=mish)
        self.mid_block2 = res_block_type(mid_dim, mid_dim * 2, mid_dim, gcond_dim=tot_cond_dim, kernel_size=self.resblock_ksize, n_groups=8, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # is_last = ind >= (num_resolutions - 1)
            is_last = ind >= (num_resolutions - 1) or ind < ( num_resolutions - self.down_times - 1)

            ##? Eg. dim_out:4, dim_in:8, dim_out*2 because we concat residual 
            self.ups.append(nn.ModuleList([
                res_block_type(dim_out * 2, dim_out, dim_in, gcond_dim=tot_cond_dim, kernel_size=self.resblock_ksize, n_groups=8, mish=mish),
                res_block_type(dim_in, dim_in * 2, dim_in, gcond_dim=tot_cond_dim, kernel_size=self.resblock_ksize, n_groups=8, mish=mish),
                Upsample1d(dim_in) if not is_last and self.use_downup_sample else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2
        
        ## -- Ordinary Diffusion Setup --
        ## FIXME: upgrade this part, adding the is inpainting feature here??
        if not self.energy_mode:
            self.final_conv = nn.Sequential(
                Conv1dBlock(base_dim, base_dim, kernel_size=self.resblock_ksize), # 5
                nn.Conv1d(base_dim, transition_dim, 1),
            )
        ## -- Energy Diffusion Parameterization Setup --
        elif self.energy_param_type == 'L2':
            raise NotImplementedError
        else:
            raise NotImplementedError()



    def forward(self, x, time,
                tj_cond: dict,
                map_feat: tuple, # map observation condition
                force_dropout=False, half_fd=False,):
        '''
            x : [ batch x horizon x transition ]
            time: [batch,]
            walls_loc: [batch, 6], 2D
            half_fd: drop the conditions for the second half in the input batch 
        '''
        
        spatial_cond, map_cond = map_feat
        
        # self.energy_mode must be False. (if true, then it'll works like potential-based diffusion model)
        if self.energy_mode:
            assert False
            x.requires_grad_(True)
            x_inp = x
        
        is_st_inpat = tj_cond['is_st_inpat'] ## torch tensor gpu
        is_end_inpat = tj_cond['is_end_inpat']
        ## sanity check
        b_size = x.shape[0]
        assert is_st_inpat.shape[0] == b_size and is_st_inpat.ndim == 1 \
            and is_st_inpat.dtype == torch.bool
        assert is_end_inpat.shape[0] == b_size and is_end_inpat.ndim == 1 \
            and is_end_inpat.dtype == torch.bool
        
        ## ----------------

        x = einops.rearrange(x, 'b h t -> b t h')

        t_feat = self.time_mlp(time) ## e.g., (B,64) ## TODO:
        
        # pdb.set_trace()

        ## Encode Wall Locations to a feature vector w
        # w = self.wallLoc_encoder(walls_loc)
        ## what we want is like [B, dim], use the cls_token if vit1d

        ## obtain feature
        st_ovlp_is_drop = tj_cond['st_ovlp_is_drop']
        end_ovlp_is_drop = tj_cond['end_ovlp_is_drop']
        # assert torch.is_tensor(st_ovlp_is_drop) and torch.is_tensor(end_ovlp_is_drop)

        if st_ovlp_is_drop is not None: ##
            st_ovlp_feat = self.st_ovlp_model(tj_cond['st_ovlp_traj'], 
                                    time=tj_cond['st_ovlp_t'])
            assert len(st_ovlp_is_drop) == len(st_ovlp_feat)
            assert st_ovlp_is_drop.dtype == torch.bool ## a numpy array
            st_ovlp_feat[ st_ovlp_is_drop ] = 0.
            # (~st_ovlp_is_drop) == 
            assert not torch.logical_and(~st_ovlp_is_drop, is_st_inpat).any() ## must be false
        else:
            ## no cond if None
            # st_ovlp_feat = torch.zeros_like(st_ovlp_feat)
            st_ovlp_feat = torch.zeros( (x.shape[0], self.st_ovlp_model.out_dim), device=x.device)

        
        if tj_cond['end_ovlp_is_drop'] is not None:
            end_ovlp_feat = self.end_ovlp_model(tj_cond['end_ovlp_traj'],
                                                time=tj_cond['end_ovlp_t'])
            end_ovlp_feat[ tj_cond['end_ovlp_is_drop'] ] = 0.
            assert end_ovlp_is_drop.dtype == torch.bool
            assert not torch.logical_and(~end_ovlp_is_drop, is_end_inpat).any()
        else:
            # end_ovlp_feat = torch.zeros_like(end_ovlp_feat)
            end_ovlp_feat = torch.zeros( (x.shape[0], self.end_ovlp_model.out_dim), device=x.device)

        ## Here we create corresponding condition feature to let the model know if we actually overwrite!
        if self.inpaint_token_type == 'const':
            # (B,token_dim)
            st_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
            num_st_inpt = torch.sum(is_st_inpat).item()
            ## assign value
            st_token[is_st_inpat] = self.st_use_inpaint_token.repeat( (num_st_inpt, 1) )
            st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat( (b_size - num_st_inpt, 1) )
            # pdb.set_trace()
            ### from Here Oct 10 14:38

            end_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
            num_end_inpt = torch.sum(is_end_inpat).item()
            end_token[is_end_inpat] = self.end_use_inpaint_token.repeat( (num_end_inpt, 1) )
            end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat( (b_size - num_end_inpt, 1) )

            st_token = self.st_inpaint_model(st_token)
            end_token = self.end_inpaint_model(end_token)
        else:
            raise NotImplementedError
        ## NOTE: for one side, we can only either do inpainting or ovlp conditioning


        if force_dropout:
            # pdb.set_trace() ## important: do not drop the st_token?
            assert not self.training
            if half_fd:
                b_s = len(st_ovlp_feat)
                # drop the second half
                assert b_s % 2 == 0
                st_ovlp_feat[int(b_s//2):] = 0. # * st_ovlp_feat[int(b_s//2):] 
                end_ovlp_feat[int(b_s//2):] = 0. # * end_ovlp_feat[int(b_s//2):] 
            else:
                assert False
                w = 0. * w

        # output of st, end overlapped feature will be treated as wall Embeddings.
        # Set this to True
        if self.cat_t_w:
            t_feat = torch.cat([t_feat, st_ovlp_feat, end_ovlp_feat, st_token, end_token, map_cond], dim=-1) # originally map_cond isn't concatenated to t_feat.
        else:
            raise Exception("cat_t_w must be True if you want to give conditionals to diffusion")
        
        h = []

        # pdb.set_trace()

        for resnet, resnet2, downsample in self.downs:
            # overlapping feature as conditionals. (we don't need to use map encoder output, observation encoder output at all ...)
            # But, the task here does not provide dataset as big as Maze2D or AntMaze. Thus conditionals could be helpful.
            x = resnet(x, t_feat) 
            x = resnet2(x, t_feat)
            h.append(x)
            x = downsample(x)

        # print(f'after downs: {x.shape}')

        x = self.mid_block1(x, t_feat)
        
        spatial_token = self.patchfier(spatial_cond)
        spa_k = self.map_k(spatial_token)
        spa_v = self.map_v(spatial_token)
        x = x.permute(0, 2, 1) # [B, H, C]
        x = torch.nn.functional.scaled_dot_product_attention(x, spa_k, spa_v) + x
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.mid_block2(x, t_feat)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t_feat)
            x = resnet2(x, t_feat)
            x = upsample(x)

        # print(f'after ups: {x.shape}')

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        if self.energy_mode:
            assert False, 'not used'

        return x
