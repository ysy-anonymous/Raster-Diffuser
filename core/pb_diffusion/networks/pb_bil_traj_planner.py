# Basic Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional Encoders, Diffusion Decoder
from core.pb_diffusion.networks.diffusion.temporal_cond_bilatent import TemporalUnet_WCond_BIL # Diffusion Trajectory Planner
from core.pb_diffusion.networks.cond_encoders.EnvEncoder import OBSEmbedEncoder # Observation ENcoder
from core.pb_diffusion.networks.cond_encoders.vit_2d import ViT # Map Encoder


# Python Typing
from typing import List, Dict


# Trajectory Planner
class PBTrajBILPlanner(nn.Module):
    """
        Potential-Based Trajectory Decoder
        network_config (Dict) file that contains the initialization information
    """
    def __init__(self, network_config):
        super().__init__()
        self.map_config = network_config['map_encoder'] 
        self.obs_config = network_config['obs_encoder']
        self.decoder_config = network_config['unet_decoder']

        self.map_encoder = ViT(**self.map_config) # map encoder
        self.obs_encoder = OBSEmbedEncoder(**self.obs_config) # observation encoder
        self.diff_decoder = TemporalUnet_WCond_BIL(**self.decoder_config) # diffusion trajectory decoder
        
        self.energy_mode = self.diff_decoder.energy_mode

    # Give scheduler information to decoder
    def set_schedule_parameters(self, betas, alphas, alphas_cumprod):
        self.diff_decoder.set_schedule_parameters(betas, alphas, alphas_cumprod)
                
    def forward(self, sample, timestep, map_cond, obs_cond,
                use_dropout=True, force_dropout=False, half_fd=False):
        
        map_encoded = self.map_encoder(map_cond)
        obs_feat = self.obs_encoder(obs_cond)
        
        noise = self.diff_decoder(sample, timestep, map_encoded, obs_feat, use_dropout=use_dropout,
                                  force_dropout=force_dropout, half_fd=half_fd)
        return noise