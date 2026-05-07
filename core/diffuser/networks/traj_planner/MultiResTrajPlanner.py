# Basic Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional Encoders, Diffusion Decoder
from core.diffuser.networks.map_encoder.ViTMapEncoder import ViT
from core.diffuser.networks.diffusion_decoder.MultiResDiffusionDecoder import MultiResDiffusionDecoder
from core.diffuser.networks.obs_encoder.EnvEncoder import OBSEmbedEncoder

# Python Typing
from typing import List, Dict

class MultiResTrajPlanner(nn.Module):
    """
        Diffusion based trajectory planner that contains conditional encoders (map encoder, observation encoder) and diffusion decoder
        network_config: configuration file that contains "map_encoder", "obs_encoder", "diffusion_decoder", "map_upsampler" as key
    """
    def __init__(self, network_config: Dict):
        super().__init__()
        self.map_config = network_config['map_encoder']
        self.obs_config = network_config['obs_encoder']
        self.decoder_config = network_config['diffusion_decoder']
        
        self.map_encoder = ViT(**self.map_config) # map encoder
        self.env_encoder = OBSEmbedEncoder(**self.obs_config) # observation encoder
        self.diff_decoder = MultiResDiffusionDecoder(**self.decoder_config) # diffusion trajectory decoder
        
        self.use_aux_loss = self.decoder_config['use_aux_loss']
        self.T_coarse = self.decoder_config['T_coarse']
        self.T_mid = self.decoder_config['T_mid']
        self.T_fine = self.decoder_config['T_fine']
    
    def set_noise_scheduler(self, noise_scheduler):
        self.diff_decoder.set_noise_scheduler(noise_scheduler)
                
    def forward(self, sample, timestep, map_cond, env_cond):
        
        mapfeat_w_tokens = self.map_encoder(map_cond)
        env_feat = self.env_encoder(env_cond)
        
        # Multi-timestep resolution residual trajectory planner
        noise, clean_traj, aux = self.diff_decoder(sample, timestep, mapfeat_w_tokens, env_feat)
        
        return noise, clean_traj, aux