# Basic Pytorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional Encoders, Diffusion Decoder
from core.diffuser.networks.map_encoder.MapEncoder import MapEncoder
from core.diffuser.networks.diffusion_decoder.MultiResDiffusionDecoder import MultiResDiffusionDecoder
from core.diffuser.networks.obs_encoder.EnvEncoder import OBSEmbedEncoder

# Python Typing
from typing import List, Dict


# Used to upsample map feature to (32, 32)
class FeatUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_ratio, upsample_method):
        super().__init__()
        self.in_channels = in_channels # number of input channels
        self.out_channels = out_channels # number of output channels
        self.upsample_ratio = upsample_ratio # upsampling ratio
        self.upsample_method = upsample_method
        if upsample_method == 0:
            if in_channels % (upsample_ratio ** 2) != 0:
                raise Exception("input channels must be divisible power of 2 of upsample_ratio.")
            self.upsampler = nn.Sequential(
                nn.PixelShuffle(upscale_factor=self.upsample_ratio),
                nn.Conv2d(in_channels=in_channels//(self.upsample_ratio ** 2), out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0, dilation=1, groups=1)
            )
        elif upsample_method == 1:
            self.upsampler = nn.ConvTranspose2d(self.in_channels, self.out_channels, 
                                                kernel_size=int(upsample_ratio), stride=int(upsample_ratio), padding=0, output_padding=0, bias=True)
        elif upsample_method == 2:
            self.upsampler = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        else:
            raise Exception("Choose one of the following methods: [(0): Pixel Unshuffle, (1): Transposed Convolution, (2): Bilinear Upsampling]")

    def forward(self, x):
        if self.upsample_method in [0, 1]:
            x = self.upsampler(x)
        elif self.upsample_method == 2:
            x = F.interpolate(x, scale_factor=self.upsample_ratio)
            x = self.upsampler(x)
                    
        return x

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
        self.upsampler_config = network_config['map_upsampler']
        
        if self.upsampler_config['in_channels'] != self.map_config['map_out']:
            print("Warning!> map feature for upsample and number of map channel output must be same!")
            self.upsampler_config['in_channels'] = self.map_config['map_out']
        self.map_upsampler = FeatUpsampler(**self.upsampler_config)
        
        self.map_encoder = MapEncoder(**self.map_config) # map encoder
        self.env_encoder = OBSEmbedEncoder(**self.obs_config) # observation encoder
        self.diff_decoder = MultiResDiffusionDecoder(**self.decoder_config) # diffusion trajectory decoder
        
        self.use_aux_loss = self.decoder_config['use_aux_loss']
        self.T_coarse = self.decoder_config['T_coarse']
        self.T_mid = self.decoder_config['T_mid']
        self.T_fine = self.decoder_config['T_fine']
        
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
        
        # Multi-timestep resolution residual trajectory planner
        noise, clean_traj, aux = self.diff_decoder(sample, timestep, map_feat_upsampled, env_feat)
        
        return noise, clean_traj, aux