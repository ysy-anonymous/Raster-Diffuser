# Basic Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional Encoders, Diffusion Decoder
# from core.networks.map_encoder.MapEncoder2 import MapEncoder
from core.diffuser.networks.map_encoder.ViTMapEncoder import ViT
from core.diffuser.networks.diffusion_decoder.BiLatentDiffusionDecoder import BiLatentFusionDiffusionTrajDecoder
from core.diffuser.networks.obs_encoder.EnvEncoder import OBSEmbedEncoder
# Python Typing
from typing import List, Dict

class BiLatentTrajectoryPlanner(nn.Module):
    """
        Bidirectional trajectory planner that predicts both trajectory and spatial map.
        Diffusion based trajectory planner that contains conditional encoders (map encoder, observation encoder) and diffusion decoder
        network_config: configuration file that contains "map_encoder", "obs_encoder", "diffusion_decoder", "bifusion_block"  as key
    """
    def __init__(self, network_config: Dict):
        super().__init__()
        self.map_config = network_config['map_encoder']
        self.obs_config = network_config['obs_encoder']
        self.decoder_config = network_config['diffusion_decoder']
        
        self.map_encoder = ViT(**self.map_config) # map encoder
        self.env_encoder = OBSEmbedEncoder(**self.obs_config) # observation encoder
        self.diff_decoder = BiLatentFusionDiffusionTrajDecoder(**self.decoder_config) # diffusion trajectory decoder

    def set_noise_scheduler(self, noise_scheduler):
        self.diff_decoder.set_noise_scheduler(noise_scheduler)
                
    def forward(self, sample, timestep, map_cond, env_cond):
        map_feat = self.map_encoder(map_cond)
        env_feat = self.env_encoder(env_cond)
        
        # Bi-directional diffusion decoder
        noise = self.diff_decoder(sample, timestep, map_feat, env_feat)
        
        return noise

        
        
        
