import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from core.diffuser.networks.map_encoder.helpers.vssm_encoder2 import VSSMEncoder

class VSSMapEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_model, dws_kernel: List, state_dim: List, ssm_exp_ratio: List, 
                 vssm_drop:List, ffn_dim: List, ffn_drop: List, num_layers: List):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding='same', dilation=1)
        )
        
        self.num_stages = len(state_dim) # number of stages
        self.encoders = nn.ModuleList([])
        for i in range(self.num_stages):
            self.encoders.append(VSSMEncoder(d_model=d_model, dws_kernel=dws_kernel[i], state_dim=state_dim[i], 
                                             ssm_exp_ratio=ssm_exp_ratio[i], vssm_drop=vssm_drop[i], ffn_dim=ffn_dim[i], ffn_drop=ffn_drop[i], num_layers=num_layers[i]))
            
        self.out_proj = nn.Conv2d(in_channels=d_model, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                  dilation=1, bias=True)
        self.avg_pooler = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, x):
        x = self.stem(x)
        for i in range(self.num_stages):
            x = self.encoders[i](x)
        x = self.out_proj(x)
        
        pooled_x = self.avg_pooler(x).flatten(1) # (B, C, 1, 1) -> (B, C)
        
        return x, pooled_x