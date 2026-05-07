# Basic Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional Encoders, Diffusion Decoder
from core.comp_diffusion.cd_stgl_sml_dfu.stgl_sml_temporal_cond_v2 import Unet1D_TjTi_Stgl_Cond_V2
from core.comp_diffusion.vit_2d import ViT # Map Encoder


# Python Typing
from typing import List, Dict


# Trajectory Planner
class CompTrajPlannerV2(nn.Module):
    """
        Potential-Based Trajectory Decoder
        network_config (Dict) file that contains the initialization information
    """
    def __init__(self, network_config):
        super().__init__()
        self.map_config = network_config['map_encoder'] 
        self.decoder_config = network_config['unet_decoder']

        self.map_encoder = ViT(**self.map_config) # map encoder
        self.diff_decoder = Unet1D_TjTi_Stgl_Cond_V2(**self.decoder_config) # diffusion trajectory decoder
        
        # self.energy_mode = self.diff_decoder.energy_mode
        self.input_t_type = self.diff_decoder.input_t_type
                
    def forward(self, sample, timestep, tj_cond, map_cond,
                force_dropout=False, half_fd=False):
        
        map_encoded = self.map_encoder(map_cond)
        
        noise = self.diff_decoder(sample, timestep, tj_cond, map_encoded, force_dropout=force_dropout, half_fd=half_fd)
        return noise