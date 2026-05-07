from utils.config_utils import get_norm_stat


################
upsample_ratio= 1.0
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

class DiffuserV1Config:
    def __init__(self):
        self.horizon = 32
        self.action_dim = 2
        self.network_config = {
            'unet_config': { # unet diffusion decoder config
                'action_dim': 2,
                'action_horizon': 32,
                'diffusion_step_embed_dim': 256,
            },
            'mlp_config': { # action encoder config
                'obs_dim': 2,
                'embed_dim': 64
            },
            'vit_config': { # vit config for map encoder
                'image_size': h, # change this if you want to use different map size
                'patch_size': 4,
                'channels': 1, # change this to 3 if you use continuous map field.
                'num_classes': 32,
                'dim': 512,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 1024,
                'pool': 'cls',
                'dropout': 0.1,
                'emb_dropout': 0.1
            },
            'cnn_config': { # cnn config for map encoder (only valid if is_CNN=True)
                'input_dim': 3, # 1 for non continuous field, 3 for continuous field
                'output_dim': 32
            } # total processing: (diffusion_step_embed_dim(256) + map_dim(32) + action_dim(64) = 352)
        }

        self.noise_scheduler = {
            "type": "ddpm",
            "ddpm": {
                "num_train_timesteps": 100,
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
            'use_ema': True,
            'batch_size': 256,
            'optimizer': {
                'name': "adamw",
                'learning_rate': 1.0e-4,
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
                
        self.is_CNN = False # whether to use CNN as map encoder, if False, use ViT

    def to_dict(self):
        return {
            "network_config": self.network_config,
            "noise_scheduler": self.noise_scheduler,
            "normalizer": self.normalizer,
            "trainer": self.trainer,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "is_CNN": self.is_CNN,
            "dataset": self.dataset
        }