import torch
import torch.nn as nn
from core.diffuser.networks.utils.helpers import LayerNorm, ChannelFirstLayerNorm
# load enums
from core.diffuser.networks.utils.enums import UPSampleStyle
# load config
from utils.config_utils import get_norm_stat

def get_map_out_channels(stem_out: int, depth):
    channel_list = [stem_out] + [stem_out * 2 ** power for power in range(depth)]
    num_channel = channel_list[-1]
    for idx in range(len(channel_list)-1, -1):
        num_channel = num_channel//2 + channel_list[idx]
    return num_channel

def main():
    map_channel_o = get_map_out_channels(64, 3)
    print("map encoder final layer output: ", map_channel_o)

if __name__ == '__main__':
    main()

dataset_upsample = 4.0 # 8 -> 32
map_feature_upsample = 2.0 # 32 -> 64
ori_map_size = (8, 8)
h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)
data_size = 2000
norm_stat = get_norm_stat(ori_map_size, data_size)

# Bi-directional Trajectory Planner Configuration
class BiLatentTrajectoryPlannerConfig:
    def __init__(self):
        self.horizon = 32
        self.action_dim = 2
        self.network_config = {
            'map_encoder': {
                'map_size': (h, w),
                'stem_in': 3, # 1 for non continous field usage, 3 for continuous field usage
                'stem_out': 64,
                'unet_out': [256, 256],
                'map_out': 512,
                'kernel_schemes': [[2, 2, 2], [4, 2, 1]],
                'stride_schemes': [[2, 2, 2], [4, 2, 1]],
                'modulator_schemes': [[3, 3, 3], [3, 3, 3]],
                'encoder_act': nn.Mish,
                'encoder_norm': LayerNorm
            },
            'obs_encoder': {
                'map_size': (h, w),
                'patch_size': (4, 4),
                'obs_dim': 4,
                'embedding_dim': 64,
                'output_dim': 128,
                'activation': nn.Mish
            },
            'diffusion_decoder': { # horizon: 32
                'stem_channels': 64,
                'channel_scheme': [128, 256, 512],
                'out_channel': 2,
                'kernel_scheme': [2, 2, 2], # ends at sequence 4
                'stride_scheme': [2, 2, 2],
                'padding_scheme': [0, 0, 0],
                'out_padding_scheme': [0, 0, 0],
                'group_scheme': [1, 1, 1], # groups per unet
                'activation': nn.Mish,
                'norm_layer': ChannelFirstLayerNorm,
                'timestep_dim': 128,
                'map_cond_dim': 256,
                'cond_dim': 512, # (256 + 128 + 128)
                'num_iter': 5,
                'upsample_size': (8, 8),
                'bilatent_dim': 64
            },
            'map_upsampler': {
                'in_channels': 512,
                'out_channels': 256,
                'upsample_ratio': float(map_feature_upsample),
                'upsample_method': UPSampleStyle.TRANSPOSE_CONV.value,
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

        # for the inference time normalization
        # if the map size changes, change this parameters accordingly
        self.normalizer = norm_stat

        self.trainer = {
            'use_ema': False, # default set to True
            'batch_size': 256,
            'optimizer': {
                'name': "adamw",
                'learning_rate': 9.0e-4, # default set to 1.0e-4
                'weight_decay': 1.0e-6
            },
            'lr_scheduler': {
                'name': "cosine",
                'num_warmup_steps': 500
            }
        }
        
        self.dataset = {
            'map_upsample_ratio': float(dataset_upsample), # input map upsapmler
            'use_continuous_field': True, # if you want to input extra continuous distance to obstacle map.
            'use_boundary_zero': True,  # use boundary zero representation
            'cell_size': 1.0, # cell size for each pixel
            'normalize': True, # normalize continous data
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