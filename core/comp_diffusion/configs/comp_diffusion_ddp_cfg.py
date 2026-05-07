import torch
import torch.nn as nn
# load configs
from utils.config_utils import get_norm_stat

dataset_upsample = 1.0 # 8 -> 8
ori_map_size = (8, 8)
h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)
# data_size = 2000
# data_size = 38345
data_size = 95792
# data_size = 1053418 
norm_stat = get_norm_stat(ori_map_size, data_size)

# ========== Overlapped Unet Version Configuration =========== #
ovlp_dim = 128
len_ovlp = 4  # default as 56, length of overlapping horizon
# st_ovlp_model, end_ovlp_model configuration
ovlp_model_config =  dict(
                        c_traj_hzn=len_ovlp,
                        in_dim=2,
                        base_dim=64, ## for cnn1d base, default as 32
                        dim_mults=(1, 4), ## default as (1, 2, 3, 4)
                        time_dim=64, ## time embedding, default as 32
                        out_dim=ovlp_dim, ## snould be same as final_mlp_dims[-1]
                        tjti_enc_config=dict(t_seq_encoder_type='mlp',
                                            cnn_out_dim=128, 
                                            final_mlp_dims=[1280, 512, ovlp_dim],
                                            f_conv_ks=3,)
                    )

# ======= horizon ======= #
horizon = 16 # for total trajectory 32, we use horizon 12


# Trajectory Planner Configuration
class CompDiffusionDDPConfig:
    def __init__(self):
        
        self.policy_config = {
            'pol_config': {
                'ev_n_comp': 3, # number of sub-segment trajectories (check with your training setup)
                'ev_top_n': 3, # top n possible trajectories (alternatives)
                'ev_pick_type': 'first' # or 'rand' -> smallest dist or randomly pick one. default as 'first'
            },
            'tj_blder_config': {
                'blend_type': 'exp',
                'exp_beta': 2
            }
        }
                
        self.diffusion_config = {
            'horizon': horizon, # horizon timestep as 12
            'observation_dim': 2,
            'action_dim': 2,
            'n_timesteps': 1000,
            'loss_type': 'l2_inv_v3', # default as l2_inv_v3
            'clip_denoised': False, # set this to true when you do reverse process
            'predict_epsilon': True,
            'action_weight' : 1.0,
            'loss_discount': 1.0,
            'loss_weights': None,
            'diff_config': {
                'obs_manual_loss_weights': {}, # must be empty dict
                'len_ovlp_cd': len_ovlp,
                'is_direct_train': True,
                'infer_deno_type': 'same',
                'w_loss_type': 'all',
                'tr_inpat_prob': 0.5,
                'tr_ovlp_prob': 0.5,
                'tr_1side_drop_prob': 0.20,
                'tr_no_ovlp_none': False, # must be False, if True 'non_repla_input_prob' must be set
                'ddim_set_alpha_to_one': True,  # default as True
                'ddim_steps' : 50, # default as 50
                'non_repla_input_prob': 0.5 # default as ?
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
            'unet_decoder': {
                'horizon': horizon, # for total 32 horizon, we use 12
                'transition_dim': 2, # location of each trajectory points
                'base_dim': 128,
                'dim_mults': (1, 4), # default as (1, 4, 8)
                'time_dim': 256,
                'map_cond_dim': 64,
                'network_config': {
                    'cat_t_w': True,
                    'resblock_ksize': 5, # currently only support 5
                    'use_downup_sample': True, # general unet
                    'st_ovlp_model_config': ovlp_model_config,
                    'end_ovlp_model_config': ovlp_model_config,
                    'ext_cond_dim': ovlp_dim * 2,
                    'ovlp_model_type': 'unet', # 'dit_enc',
                    'inpaint_token_dim': 64, # default as 32
                    'inpaint_token_type': 'const',
                    'energy_mode': False, # in comp_diffusion, must be false
                    'last_conv_ksize': 1, # 1 is best case (from diffuser)
                    'force_residual_conv': False, # must be false
                    'time_mlp_config': 3, # same as setting as False
                    'down_times': 3  # number of downsampling times
                },
            }
        }

        self.normalizer = norm_stat
        
        self.trainer = {
            'ema_decay': 0.995, 
            'train_batch_size': 256, # 128 as default
            'train_lr': 4e-4, # 2e-4 as default
            'gradient_accumulate_every': 1,
            'step_start_ema': 4000, # default as 2000
            'update_ema_every': 10,
            'log_freq': 100,
            'sample_freq': 1000, # ori: 1000
            'save_freq': 100000, # save frequency, ori:1000
            'label_freq': 1000, #
            'horizon': horizon, # for total 32 horizon, we use 12
            'n_reference': 40, ## checkparam
            'n_samples': 10, ## checkparam
            'device': 'cuda:1',
            'results_folder': f'/exhdd/seungyu/diffusion_motion/trained_weights/comp_diffusion_ddp_{data_size}/run1',
            'n_train_steps': 201000, ## new args -> number of training steps
            'trainer_dict': {
                'use_amp': True,
                'amp_dtype': 'fp16',
                'clip_grad_norm': 1.0,
                'lr_scheduler': {
                    'name': "cosine",
                    'num_warmup_steps': 4000, # default set as 500 for 100 training dataset
                    'num_cycles': 0.60 # 1.0 means complete 1 cycle of consine
                }
            }
        }
        
        self.dataset = {
            'horizon': horizon, # horizon of truncated traj segments
            'stride': 4, # window stride that performs truncation
            'include_map': True, # must be true
            'normalize_mode': 'dataset',
            'coord_min': 0.0,
            'coord_max': 8.0,
            'include_distance_map': True,
            'cell_size': 1.0,
            'boundary_zero': True,
            'input_upsample': dataset_upsample
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