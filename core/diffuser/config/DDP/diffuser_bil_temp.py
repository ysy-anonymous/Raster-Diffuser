import torch.nn as nn
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
# data_size = 97529 # note that this one has horizon length 64, not 32

norm_stat = get_norm_stat(ori_map_size, data_size)

# Bi-directional Trajectory Planner Configuration
class DiffuserBilDDPConfig:
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
                'in_channels': 3, # 3 for include_distance_map=True, sdf_only=False settings. otherwise 1.
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
                'activation': nn.GELU
            },
            'diffusion_decoder': { # horizon: 32
                'stem_channels': 64,
                'channel_scheme': [256, 384, 512],
                'out_channel': 2,
                'kernel_scheme': [2, 2, 1], # ends at sequence 4
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
                'patch_size': (4, 4),
                'num_iter': 3, # important hyper parameters (3 as default)
                'upsample_size': (8, 8), # same as input map size as default settings
                'bilatent_dim': 64, # 64 as default
                'bilatent_sigma': 0.1, # 1.2 as default value, smaller value yields more local footprint of trajectory.
                
                ## Below are set to default values. You don't need to change them unless you want to do ablation studies on the bi-latent fusion block.
                'bilatent_update_coeff': 0.2, # 0.2 as default value, smaller value means more conservative update from bi-latent fusion block
                'bilatent_spatial_blocks': 3, # number of residual blocks in spatial refine module. 3 as default
                'bilatent_correct_dim': 128, # dimension of the hidden layer in the point corrector. 128 as default
                'bilatent_mode': 'raster_w_velocity', # 'raster_only' or 'raster_w_velocity'. Internally set cp==1 or cp==3. default as 'raster_w_velocity'
                'bilatent_weight_sharing': True # whether to share weights for the spatial refine and point corrector blocks across iterations. Default as True.
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

        # for the inference time normalization
        # if the map size changes, change this parameters accordingly
        self.normalizer = norm_stat

        self.trainer = {
            'batch_size': 256,
            'num_workers': 2,
            'pin_memory': True,
            'drop_last': False,
            'use_ema': True, # default set to True
            
            'use_amp': True,
            'amp_dtype' : "fp16", # or 'bf16'
            
            'optimizer': {
                'name': "adamw",
                'learning_rate': 8.0e-4, # default set to 1.0e-4
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




import torch.nn as nn
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
# ori_map_size = (8, 8)
ori_map_size = (16, 16)

h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)

# Choose Below One of the data size (for 8x8 map size)
# data_size = 2000
# data_size = 38345
# data_size = 95792
# data_size = 1053418

# Choose Below One of the data size (for 16x16 map size)
# data_size = 6257
data_size = 97529 # note that this one has horizon length 64, not 32

# Choose Below One of the data size (for 32x32 map size)
# data_size = 11210
# data_size = 95035 # note that this one has horizon length 128, not 32, 64

norm_stat = get_norm_stat(ori_map_size, data_size)

# Bi-directional Trajectory Planner Configuration
class DiffuserBilDDPConfig:
    def __init__(self):
        self.horizon = 64 # 64 for 16x16 with 97529 data size
        self.action_dim = 2
        self.network_config = {
            'map_encoder': {
                'image_size': (h, w),
                'patch_size': (4, 4),
                'stem_channels': 32,
                'cond_out_dim': 128, # cls or mean token output dim, default as 64
                'featmap_out_dim': 128, # feature map output channels, default as 64
                'dim': 512,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 1024,
                'pool': 'cls',
                'in_channels': 1, # 3 for include_distance_map=True, sdf_only=False settings. otherwise 1.
                'dim_head': 64,
                'dropout': 0.1,
                'emb_dropout': 0.1
            },
            'obs_encoder': {
                'map_size': (h, w),
                'patch_size': (4, 4),
                'obs_dim': 4,
                'embedding_dim': 64,
                'output_dim': 128, # output dimension of obs_encoder, default as 64
                'activation': nn.GELU
            },
            'diffusion_decoder': { # horizon: 32
                'stem_channels': 64,
                'channel_scheme': [256, 384, 512],
                'out_channel': 2,
                'kernel_scheme': [2, 2, 1], # ends at sequence 4 for 32 horizon
                'stride_scheme': [2, 2, 1],
                'padding_scheme': [0, 0, 0],
                'out_padding_scheme': [0, 0, 0],
                'group_scheme': [1, 1, 1], # groups per unet
                'activation': nn.Mish,
                'norm_layer': nn.GroupNorm, # ChannelFirstLayerNorm
                'n_groups': 8, # number of groups of GroupNorm
                'timestep_dim': 256,
                'map_cond_dim': 128, # default as 64, see the map_encoder's cond_out_dim, featmap_out_dim
                'cond_dim': 512, # default as 384 (64 + 64 + 256)
                'map_size': (16, 16),
                'patch_size': (4, 4),
                'num_iter': 3, # important hyper parameters (3 as default)
                'upsample_size': (16, 16), # same as input map size as default settings
                'bilatent_dim': 128, # 64 as default
                'bilatent_sigma': 1.2, # 1.2 as default value, smaller value yields more local footprint of trajectory.
                
                ## Below are set to default values. You don't need to change them unless you want to do ablation studies on the bi-latent fusion block.
                'bilatent_update_coeff': 0.2, # 0.2 as default value, smaller value means more conservative update from bi-latent fusion block
                'bilatent_spatial_blocks': 3, # number of residual blocks in spatial refine module. 3 as default
                'bilatent_correct_dim': 128, # dimension of the hidden layer in the point corrector. 128 as default
                'bilatent_mode': 'raster_only', # 'raster_only' or 'raster_w_velocity'. Internally set cp==1 or cp==3. default as 'raster_w_velocity'
                'bilatent_weight_sharing': True # whether to share weights for the spatial refine and point corrector blocks across iterations. Default as True.
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

        # for the inference time normalization
        # if the map size changes, change this parameters accordingly
        self.normalizer = norm_stat

        self.trainer = {
            'batch_size': 256,
            'num_workers': 2,
            'pin_memory': True,
            'drop_last': False,
            'use_ema': True, # default set to True
            
            'use_amp': True,
            'amp_dtype' : "fp16", # or 'bf16'
            
            'optimizer': {
                'name': "adamw",
                'learning_rate': 6.0e-4, # default set to 1.0e-4
                'weight_decay': 1.0e-6
            },
            'lr_scheduler': {
                'name': "cosine",
                'num_warmup_steps': 4000, # default set as 500 for 100 training dataset
                'num_cycles': 0.75 # 1.0 means complete 1 cycle of consine
            }
        }
        
        self.dataset = {
            'include_distance_map': False, # default as True
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





import torch.nn as nn
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
# ori_map_size = (8, 8)
# ori_map_size = (16, 16)
ori_map_size = (32, 32)

h = int(ori_map_size[0] * dataset_upsample); w = int(ori_map_size[1] * dataset_upsample)

# Choose Below One of the data size (for 8x8 map size)
# data_size = 2000
# data_size = 38345
# data_size = 95792
# data_size = 1053418

# Choose Below One of the data size (for 16x16 map size)
# data_size = 6257
# data_size = 97529 # note that this one has horizon length 64, not 32

# Choose Below One of the data size (for 32x32 map size)
# data_size = 11210
data_size = 95035 # note that this one has horizon length 128, not 32, 64

norm_stat = get_norm_stat(ori_map_size, data_size)

# Bi-directional Trajectory Planner Configuration
class DiffuserBilDDPConfig:
    def __init__(self):
        self.horizon = 128 # 64 for 16x16 with 97529 data size, 128 for 32x32 with 95035 data size
        self.action_dim = 2
        self.network_config = {
            'map_encoder': {
                'image_size': (h, w),
                'patch_size': (4, 4),
                'stem_channels': 32,
                'cond_out_dim': 128, # cls or mean token output dim, default as 64
                'featmap_out_dim': 128, # feature map output channels, default as 64
                'dim': 512,
                'depth': 6,
                'heads': 8,
                'mlp_dim': 1024,
                'pool': 'cls',
                'in_channels': 1, # 3 for include_distance_map=True, sdf_only=False settings. otherwise 1.
                'dim_head': 64,
                'dropout': 0.1,
                'emb_dropout': 0.1
            },
            'obs_encoder': {
                'map_size': (h, w),
                'patch_size': (4, 4),
                'obs_dim': 4,
                'embedding_dim': 64,
                'output_dim': 128, # output dimension of obs_encoder, default as 64
                'activation': nn.GELU
            },
            'diffusion_decoder': { # horizon: 32
                'stem_channels': 64,
                'channel_scheme': [128, 256, 384, 512],
                'out_channel': 2,
                'kernel_scheme': [2, 2, 2, 1], # ends at sequence 4 for 32 horizon
                'stride_scheme': [2, 2, 2, 1],
                'padding_scheme': [0, 0, 0, 0],
                'out_padding_scheme': [0, 0, 0, 0],
                'group_scheme': [1, 1, 1, 1], # groups per unet
                'activation': nn.Mish,
                'norm_layer': nn.GroupNorm, # ChannelFirstLayerNorm
                'n_groups': 8, # number of groups of GroupNorm
                'timestep_dim': 256,
                'map_cond_dim': 128, # default as 64, see the map_encoder's cond_out_dim, featmap_out_dim
                'cond_dim': 512, # default as 384 (64 + 64 + 256) for 8x8, 512 (128 + 128 + 256) for 16x16
                'map_size': (32, 32),
                'patch_size': (4, 4),
                'num_iter': 3, # important hyper parameters (3 as default)
                'upsample_size': (32, 32), # same as input map size as default settings
                'bilatent_dim': 256, # 64 for 8x8, 128 for 16x16, 256 for 32x32
                'bilatent_sigma': 1.2, # 1.2 as default value, smaller value yields more local footprint of trajectory.
                
                ## Below are set to default values. You don't need to change them unless you want to do ablation studies on the bi-latent fusion block.
                'bilatent_update_coeff': 0.2, # 0.2 as default value, smaller value means more conservative update from bi-latent fusion block
                'bilatent_spatial_blocks': 3, # number of residual blocks in spatial refine module. 3 as default
                'bilatent_correct_dim': 128, # dimension of the hidden layer in the point corrector. 128 as default
                'bilatent_mode': 'raster_only', # 'raster_only' or 'raster_w_velocity'. Internally set cp==1 or cp==3. default as 'raster_w_velocity'
                'bilatent_weight_sharing': True # whether to share weights for the spatial refine and point corrector blocks across iterations. Default as True.
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

        # for the inference time normalization
        # if the map size changes, change this parameters accordingly
        self.normalizer = norm_stat

        self.trainer = {
            'batch_size': 256,
            'num_workers': 2,
            'pin_memory': True,
            'drop_last': False,
            'use_ema': True, # default set to True
            
            'use_amp': True,
            'amp_dtype' : "fp16", # or 'bf16'
            
            'optimizer': {
                'name': "adamw",
                'learning_rate': 6.0e-4, # default set to 1.0e-4
                'weight_decay': 1.0e-6
            },
            'lr_scheduler': {
                'name': "cosine",
                'num_warmup_steps': 4000, # default set as 500 for 100 training dataset
                'num_cycles': 0.75 # 1.0 means complete 1 cycle of consine
            }
        }
        
        self.dataset = {
            'include_distance_map': False, # default as True
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
    
    