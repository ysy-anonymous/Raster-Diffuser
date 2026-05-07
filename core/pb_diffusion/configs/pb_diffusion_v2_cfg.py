import torch
import torch.nn as nn
# load configs
from utils.config_utils import get_norm_stat

dataset_upsample = 1.0 # 8 -> 8
ori_map_size = (8, 8)
h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)
#data_size = 2000
data_size = 38345
# data_size = 95792
# data_size = 1053418
norm_stat = get_norm_stat(ori_map_size, data_size)

# Trajectory Planner Configuration
class PBDiffusionV2Config:
    def __init__(self):
        
        self.policy_config = {
            'seq_eval': False,
            'depoch_list' : [int(19e5), ],
            'ddim_steps': 8,
            'n_prob_env': 20,
            'use_ddim': True, # Whether to use ddim policy for reverse process
            'cond_w' : 2.0, # classifier-free guidance cond weights
            'do_replan': False, # whether to replan
            'samples_perprob' : 10
        }
                
        self.diffusion_config = {
            'horizon': 32,
            'observation_dim': 2,
            'action_dim': 2,
            'n_timesteps': 100,
            'loss_type': 'l2',
            'clip_denoised': True, # set this to true when you do reverse process
            'predict_epsilon': True,
            'loss_discount': 1.0,
            'loss_weights': None,
            'condition_guidance_w': 0.1,
            'diff_config': {
                'train_apply_condition': True,
                'set_cond_noise_to_0': False,
                'debug_mode': False,
                'manual_loss_weights': {0: 0.0, -1: 0.0},
                'ddim_steps': 10,
                'ddim_set_alpha_to_one': True,
                'is_dyn_env': False
                }
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
                'map_size': (8, 8),
                'patch_size': (4, 4),
                'horizon': 32, # our task uses horizon=32
                'transition_dim': 2, # location of each trajectory points
                'time_dim': 128,
                'map_dim': 64,
                'obs_dim': 64,
                'dim': 64, # default as 64
                'dim_mults': (1, 4, 8), # default as (1, 4, 8)
                'network_config': {
                    'cat_t_w': True,
                    'resblock_ksize': 5, # currently only support 5
                    'use_downup_sample': True, # general unet
                    'energy_mode': True, # use energy-based reverse process
                    'energy_param_type': 'L2',
                    'conv_zero_init': False,
                    'concept_drop_prob': 0.2,
                    'last_conv_ksize': 1, # 1 is best case (from diffuser)
                    'force_residual_conv': False, # must be false
                    'time_mlp_config': 1, # same as setting as False
                    'down_times': 3
                },
            }
        }

        self.normalizer = norm_stat

        self.trainer = {
            'ema_decay': 0.995, # default as 0.995, 0.0 -> no ema
            'train_batch_size': 512,
            'train_lr': 2e-4,
            'gradient_accumulate_every': 1,
            'step_start_ema': 4000, # default as 2000
            'update_ema_every': 10,
            'log_freq': 100,
            'sample_freq': 1000, # ori: 1000
            'save_freq': 1029000, # save frequency, ori:1000
            'label_freq': 1000, #
            'save_parallel': False,
            'n_reference': 50, ## checkparam
            'bucket': None,
            'train_device': 'cuda:0',
            'save_checkpoints': True,
            'results_folder': '/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_v2_1M/run1',
            'n_samples': 10, ## checkparam
            'num_render_samples': 2, ##
            'clip_grad': False,
            'clip_grad_max': False,
            'n_train_steps': 4120000,
            'trainer_dict': {
                'lr_warmupDecay': False,
                'warmup_steps_pct': 0.2, # only used if lr_warmupDecay is True
                'cycle_ratio': 1.0 # use cosine full cycle when 1.0
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