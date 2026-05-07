import torch
import torch.nn as nn
# load configs
from utils.config_utils import get_norm_stat

dataset_upsample = 1.0 # 8 -> 8
ori_map_size = (8, 8)
h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)
data_size = 2000
# data_size = 38345
# data_size = 95792
# data_size = 1053418
norm_stat = get_norm_stat(ori_map_size, data_size)

# Trajectory Planner Configuration
class ReDiffConfig:
    def __init__(self):
        
        self.policy_config = {
            'horizon': 32, # default as 32, set this to your training horizon
            'cool_term': 1.0, # default as 1.0, set to smaller value for more stochasticity
        }
                
        self.diffusion_config = {
            'horizon': 32,
            'observation_dim': 2,
            'n_timesteps': 100,
            'loss_type': 'l2',
            'clip_denoised': True, # set this to true when you do reverse process
            'predict_epsilon': True,
            'observation_weight': 1.0,
            'loss_discount': 1.0,
            'loss_weights': None,
        }
        
        self.network_config = {
            'map_encoder': {
                'image_size': (h, w),
                'patch_size': (4, 4),
                'stem_channels': 32,
                'cond_out_dim': 64, # cls or mean token output dim
                'featmap_out_dim': 64, # feature map output channels,
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
            'obs_encoder': { # currently no positional encoding is used.. (Change the parameter later)
                'map_size': (h, w),
                'patch_size': (4, 4),
                'obs_dim': 4,
                'embedding_dim': 64,
                'output_dim': 64,
                'activation': nn.Mish
            },
            'unet_decoder': { # horizon: 32
                'horizon': 32, # our task uses horizon=32
                'transition_dim': 2, # location of each trajectory points
                'map_cdim': 64,
                'obs_cdim': 64,
                'time_cdim': 128,
                'map_size': (8, 8),
                'patch_size': (4, 4),
                'dim': 128, # default as 64
                'dim_mults': (1, 2, 4), # default as (1, 4, 8)
            }
        }

        self.normalizer = norm_stat

        self.trainer = {
            'ema_decay': 0.995, # default as 0.995, 0.0 -> no ema
            'train_batch_size': 256,
            'train_lr': 3e-4,
            'gradient_accumulate_every': 1,
            'step_start_ema': 4000, # default as 2000
            'update_ema_every': 10,
            'log_freq': 100,
            'save_freq': 10000, # save frequency, ori:1000
            'label_freq': 1000, #
            'save_parallel': False,
            'results_folder': '/exhdd/seungyu/diffusion_motion/trained_weights/rediffuser/run1',
            'bucket': None,
            'n_train_steps': 21000,
            'train_device': 'cuda:4',
            'trainer_dict': {
                'lr_scheduler': {
                    'name': 'cosine',
                    'num_warmup_steps': 4000,
                    'num_cycles': 1.0
                }
            }
        }
        
        self.dataset = {
            'include_distance_map': True,
            'cell_size': 1.0,
            'boundary_zero': True,
            'input_upsample': 1.0,
        }

    def to_dict(self):
        return {
            "network_config": self.network_config,
            "diffusion_config": self.diffusion_config,
            "policy_config": self.policy_config,
            "normalizer": self.normalizer,
            "trainer": self.trainer,
            'dataset': self.dataset
        }