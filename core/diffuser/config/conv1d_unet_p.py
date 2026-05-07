from utils.config_utils import get_norm_stat

################
upsample_ratio= 1.0
ori_map_size = (8, 8)
image_size = int(upsample_ratio * ori_map_size[0])
data_size = 100
norm_stat = get_norm_stat(ori_map_size, data_size)

class Conv1dUnetTransP:
    def __init__(self):
        self.horizon = 32
        self.action_dim = 2
        self.network_config = {
            'mlp_config': { # action encoder config (MLP layer: 2 -> 256 -> 64 dim with mish activation)
                'obs_dim': 2,
                'embed_dim': 64
            },
            'vit_config': { # vit config for map encoder (ViT Layers: self-attention and project to lower dimension(=num_classes))
                'image_size': image_size,
                'patch_size': 4,
                'channels': 1, # change this to 3 if you use continuous map field.
                'num_classes': 64, # should have same dimension as mlp_config 'embed_dim'
                'dim': 512,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 1024,
                'pool': 'none', # you can use 'mean' pooling or 'cls' token
                'dropout': 0.1,
                'emb_dropout': 0.1
            },
            'cnn_config': { # cnn config for map encoder (only valid if is_CNN=True)
                'input_dim': 3, # 1 for non continuous input, 3 for continuous input
                'output_dim': 64
            },
            'latent_transformer_config': {
                'depth': 2,
                'heads': 4,
                'mlp_dim': 1024,
                'dropout': 0.1,
            },
            'unet_config': { # unet diffusion decoder config
                'action_dim': 2,
                'action_horizon': 32,
                'condition_dim': 256,
                'down_dims': [256, 512, 1024],
                'kernel_size': 5,
                'n_groups': 8,
            },
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
