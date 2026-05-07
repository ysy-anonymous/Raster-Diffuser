# Basic Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional Encoders, Diffusion Decoder
from core.diffuser.networks.map_encoder.vssmMapEncoder2 import VSSMapEncoder
from core.diffuser.networks.diffusion_decoder.BiLatentMultiDecoder import BiLatentMultiHypoDiffTrajDecoder
from core.diffuser.networks.obs_encoder.EnvEncoder import OBSEmbedEncoder
from core.diffuser.networks.utils.helpers import FeatUpsampler

# Python Typing
from typing import List, Dict

class TrajectoryPlannerPlus(nn.Module):
    """
        Diffusion based trajectory planner that contains conditional encoders (map encoder, observation encoder) and diffusion decoder
        network_config: configuration file that contains "map_encoder", "obs_encoder", "diffusion_decoder", "map_upsampler" as key
    """
    def __init__(self, network_config: Dict):
        super().__init__()
        self.map_config = network_config['map_encoder']
        self.obs_config = network_config['obs_encoder']
        self.decoder_config = network_config['diffusion_decoder']
        self.upsampler_config = network_config['map_upsampler']
        
        if self.upsampler_config['in_channels'] != self.map_config['out_channels']:
            print("Warning!> map feature for upsample and number of map channel output must be same!")
            self.upsampler_config['in_channels'] = self.map_config['out_channels']
        self.map_upsampler = FeatUpsampler(**self.upsampler_config)
        
        self.map_encoder = VSSMapEncoder(**self.map_config) # map encoder
        self.env_encoder = OBSEmbedEncoder(**self.obs_config) # observation encoder
        self.diff_decoder = BiLatentMultiHypoDiffTrajDecoder(**self.decoder_config) # diffusion trajectory decoder
        
        self.init_weight()
    
    def set_noise_scheduler(self, noise_scheduler):
        self.diff_decoder.set_noise_scheduler(noise_scheduler)
    
    def init_weight(self):
        def _init_upsampler(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                if self.map_config['encoder_act'] in [nn.ReLU, nn.LeakyReLU]:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
                    
        self.map_upsampler.apply(_init_upsampler)
                
    def forward(self, sample, timestep, map_cond, env_cond):
        
        map_feat = self.map_encoder(map_cond)
        env_feat = self.env_encoder(env_cond)
        
        map_feat_upsampled = self.map_upsampler(map_feat)
        
        # Multi-Hypothesis based diffusion decoder
        noise, clean_traj = self.diff_decoder(sample, timestep, map_feat_upsampled, env_feat)
        # noise, clean_traj, logits = self.diff_decoder(sample, timestep, map_feat_upsampled, env_feat)
        return noise, clean_traj
        # return noise, clean_traj, logits

        
        
        
