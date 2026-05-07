import torch
import torch.nn as nn

from core.diffuser.networks.utils.helpers import LayerNorm, ChannelFirstLayerNorm

# load config
from utils.config_utils import get_norm_stat


dataset_upsample = 1.0 # 8 -> 8
ori_map_size = (8, 8)
# ori_map_size = (16, 16)

h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)

# Choose Below One of the data size (for 32x32 map size)
# data_size = 2000
data_size = 38345
# data_size = 95792
# data_size = 1053418

# Choose Below One of the data size (for 64x64 map size)
# data_size = 6257
# data_size = 97529

norm_stat = get_norm_stat(ori_map_size, data_size)

# Trajectory Planner Configuration
class DiffuserV2Config:
    def __init__(self):
        self.horizon = 32
        self.action_dim = 2
        self.network_config = {
            'map_encoder': {
                'image_size': (h, w),
                'patch_size': (4, 4),
                'stem_channels': 32,
                'cond_out_dim': 64, # cls or mean token output dim
                'featmap_out_dim': 64, # feature map output channels
                'dim': 512,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 1024,
                'pool': 'cls',
                'in_channels': 3,
                'dim_head': 64,
                'dropout': 0.1,
                'emb_dropout': 0.1
            },
            'obs_encoder': {
                'map_size': (h, w),
                'patch_size': (4, 4),
                'obs_dim': 4,
                'embedding_dim': 64,
                'output_dim': 64,
                'activation': nn.Mish
            },
            'diffusion_decoder': { # horizon: 32
                'stem_channels': 64,
                'channel_scheme': [256, 384, 512],
                'out_channel': 2,
                'kernel_scheme': [2, 2, 1], # ends at sequence 8
                'stride_scheme': [2, 2, 1],
                'padding_scheme': [0, 0, 0],
                'out_padding_scheme': [0, 0, 0],
                'group_scheme': [1, 1, 1], # groups per unet
                'activation': nn.Mish,
                'norm_layer': nn.GroupNorm, # ChannelFirstLayerNorm
                'n_groups': 8, # number of groups of GroupNorm
                'timestep_dim': 256,
                'map_cond_dim': 64,
                'cond_dim': 384, # (64 + 64 + 256)
                'map_size': (8, 8),
                'patch_size': (4, 4)
            },
        }

        self.noise_scheduler = {
            "type": "ddpm",
            "ddpm": {
                "num_train_timesteps": 100, # default as 100
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "prediction_type": "epsilon",
            },
            "ddim": {
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "prediction_type": "epsilon",
            },
            "dpmsolver": {
                "num_train_timesteps": 100,
                "beta_schedule": "squaredcos_cap_v2",
                "prediction_type": "epsilon",
                "use_karras_sigmas": True,
            },
        }

        self.normalizer = norm_stat
        
        self.trainer = {
            'use_ema': True, # default set to True
            'batch_size': 256,
            'optimizer': {
                'name': "adamw",
                'learning_rate': 3.0e-4, # default set to 1.0e-4
                'weight_decay': 1.0e-6
            },
            'lr_scheduler': {
                'name': "cosine",
                'num_warmup_steps': 4000, # default set as 500 for 100 training dataset
                'num_cycles': 1.0 # 1.0 means complete 1 cycle of consine
            }
        }
        
        self.dataset = {
            'include_distance_map': True, # default as True
            'sdf_only': False, # default as False, if True, only returns SDF map. note that include_distance_map must be True also. 
            'dmap_only': False, # default as False. Only set one of sdf_only or dmap_only to True, not both.
            'cell_size': 1.0,
            'boundary_zero': True,
            'input_upsample': 1.0,
        }

    def to_dict(self):
        return {
            "network_config": self.network_config,
            "noise_scheduler": self.noise_scheduler,
            "normalizer": self.normalizer,
            "trainer": self.trainer,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            'dataset': self.dataset
        }