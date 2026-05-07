import torch
import torch.nn as nn
from core.diffuser.networks.utils.helpers import LayerNorm, ChannelFirstLayerNorm
# load enums
from core.diffuser.networks.utils.enums import UPSampleStyle
# load configs
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

dataset_upsample = 1.0 # 8 -> 8
map_feature_upsample = 1.0 # 8 -> 8
ori_map_size = (8, 8)
h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)
data_size = 2000
norm_stat = get_norm_stat(ori_map_size=ori_map_size, data_size=data_size)

# Trajectory Planner Configuration
class MultiHypothesisTrajectoryPlannerConfig:
    def __init__(self):
        self.horizon = 32
        self.action_dim = 2
        self.network_config = {
            'map_encoder': {
                'in_channels': 3,
                'out_channels': 256,
                'input_size': (h, w),
                'd_model': 256,
                'dws_kernel': 3,
                'state_dim': [64, 64, 64, 64],
                'ssm_exp_ratio': [1, 1, 1, 1],
                'vssm_drop': 0.0,
                'patch_size': [(1, 1), (1, 1), (1, 1), (1, 1)],
                'attn_drop': 0.1,
                'num_heads': 4,
                'use_patch_proj': False,
                'ffn_dim': 1024,
                'ffn_dropout': 0.1,
                'num_layers': [3, 3, 3, 3]
            },
            'obs_encoder': {
                'map_size': (h, w),
                'patch_size': (4, 4),
                'obs_dim': 4,
                'embedding_dim': 64,
                'output_dim': 128,
                'activation': nn.GELU
            },
            'diffusion_decoder': { # horizon: 32
                'stem_channels': 64,
                'channel_scheme': [128, 256, 512],
                'kernel_scheme': [2, 2, 2], # ends at sequence 4
                'stride_scheme': [2, 2, 2],
                'padding_scheme': [0, 0, 0],
                'out_padding_scheme': [0, 0, 0],
                'group_scheme': [1, 1, 1], # groups per unet
                'activation': nn.Mish,
                'norm_layer': ChannelFirstLayerNorm,
                'timestep_dim': 128,
                'map_cond_dim': 256,
                'obs_dim': 128,
                'num_hypothesis': 8,
                'token_dim': 256,
                'hidden_dim': 256,
                'num_iters': 3
            },
            'map_upsampler': {
                'in_channels': 256,
                'out_channels': 256,
                'upsample_ratio': float(map_feature_upsample),
                'upsample_method': UPSampleStyle.TRANSPOSE_CONV.value,
            }
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
            'use_ema': False, # default set to True
            'batch_size': 128,
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
            'map_upsample_ratio': float(dataset_upsample), # input map upsampler
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