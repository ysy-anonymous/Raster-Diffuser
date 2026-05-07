# Basic Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional Encoders, Diffusion Decoder
from core.rediffuser.networks.diffuser.models.temporal import TemporalUnet
from core.rediffuser.networks.cond_encoders.EnvEncoder import OBSEmbedEncoder
from core.rediffuser.networks.cond_encoders.vit_2d import ViT # map Encoder


# Trajectory Planner
class RediffuserPlanner(nn.Module):
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
        self.diff_decoder = TemporalUnet(**self.decoder_config) # diffusion trajectory decoder
        
                
    def forward(self, sample, timestep, map_cond, obs_cond):
        
        map_encoded = self.map_encoder(map_cond)
        obs_feat = self.obs_encoder(obs_cond)
        
        noise = self.diff_decoder(sample, timestep, map_encoded, obs_feat)
        return noise