import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from core.diffuser.networks.map_encoder.helpers.vssm_encoder import VSSMEncoder


class VSSMapEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, d_model, dws_kernel, state_dim: List, ssm_exp_ratio: List, vssm_drop,
                         patch_size: List, attn_drop, num_heads, use_patch_proj, ffn_dim, ffn_dropout, num_layers: List):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding='same', dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding='same', dilation=1)
        )
        self.num_stages = len(patch_size)
        self.vssm_encoders = nn.ModuleList([])
        for i in range(self.num_stages):
            self.vssm_encoders.append(VSSMEncoder(input_size=input_size, d_model=d_model, dws_kernel=dws_kernel, state_dim=state_dim[i],
                                                  ssm_exp_ratio=ssm_exp_ratio[i], vssm_drop=vssm_drop, 
                                                  patch_size=patch_size[i], attn_drop=attn_drop, num_heads=num_heads, use_patch_proj=use_patch_proj,
                                                  ffn_dim=ffn_dim, ffn_dropout=ffn_dropout, num_layers=num_layers[i]))    
        self.out_proj = nn.Conv2d(in_channels=d_model, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                  dilation=1, bias=True)
        self.avg_pooler = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.num_stages):
            x = self.vssm_encoders[i](x) # apply vssm encoders
        x = self.out_proj(x) # output projection
        
        pooled_x = self.avg_pooler(x).flatten(1) # (B, C, 1, 1) -> (B, C)
               
        return x, pooled_x