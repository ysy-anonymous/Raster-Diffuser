import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

# import VSSM
from core.diffuser.networks.map_encoder.helpers.segman_encoder import VSSM, LayerNorm2d

# import Patchfier
from core.diffuser.networks.utils.helpers import Patchfier, UnPatchfier
from core.diffuser.networks.utils.enums import PatchifyStyle


class DWSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depth_convolution = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                                           stride=1, padding=kernel_size//2, bias=True, groups=in_channels)
        self.pw_convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x = self.depth_convolution(x)
        x = self.pw_convolution(x)
        return x


# VSSM Based mixer (mostly token-mixer, few channel-mixer)
class VSSMMixer(nn.Module):
    
    def __init__(self, input_size, d_model, dws_kernel=3, state_dim=1, ssm_exp_ratio=1, vssm_drop=0.0, patch_size=(4, 4), attn_drop=0.0, num_heads=8, use_patch_proj=False):
        super().__init__()
        
        # dws conv before ssm (Non-Linear)
        self.pre_dws = DWSConv(in_channels=d_model, out_channels=d_model, kernel_size=dws_kernel)
        self.pre_act = nn.GELU()
        
        # Token-mixer (Linear but LTV)
        self.ssm = VSSM(d_model=d_model, d_state=state_dim, expansion_ratio=ssm_exp_ratio, dropout=vssm_drop) # VSSM Layer
        self.norm = LayerNorm2d(d_model) # Normalization Layer
        
        # dws conv after ssm (Non-Linear)
        self.post_dws = DWSConv(in_channels=d_model, out_channels=d_model*3, kernel_size=dws_kernel) # Depth-wise Seperable Convolution
        
        # Patchfier
        self.patchfier = Patchfier(input_size=input_size, patch_size=patch_size, style=PatchifyStyle.VIT_STYLE.value)
        self.unpatchfier = UnPatchfier(input_size=input_size, patch_size=patch_size, feat_channels=d_model)
        
        self.use_patch_proj = use_patch_proj
        if use_patch_proj:
            # Patch Projection
            self.patch_proj_in = nn.Linear(in_features=d_model * patch_size[0] * patch_size[1] * 3, out_features=d_model * 3, bias=True)
            self.patch_proj_out = nn.Linear(in_features=d_model, out_features=d_model*patch_size[0]*patch_size[1], bias=True)
            
        # Standard Multi-Head Attention
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=d_model * patch_size[0] * patch_size[1], num_heads=num_heads, dropout=attn_drop, batch_first=True)
    
    def forward(self, x):
        """
        x: (B, C, H, W) 2D spatial tensor
        """
        B, C, H, W = x.shape
        input = x
        x = self.pre_dws(x)
        x = self.pre_act(x)
        
        # Output of SSM does not output 2D structured tensor.
        # See the segman_encoder VSSM Class for details.
        x = self.ssm(x)
        x = self.norm(x.reshape(B, -1, H, W))
        
        qkv = self.post_dws(x) # (B, C * 3, H, W)
        qkv = self.patchfier(qkv) # (B, L, C * 3 * patch_h * patch_w)
        if self.use_patch_proj:
            qkv = self.patch_proj_in(qkv) # (B, L, C * 3)
        q,k,v = torch.chunk(qkv, chunks=3, dim=-1) # (B, L, C) for each q,k,v
        attn_o, _ = self.multi_head_attn(q, k, v)
        if self.use_patch_proj:
            attn_o = self.patch_proj_out(attn_o) # (B, L, C * patch_h * patch_w)
        
        # restore spatial structure
        spatial_feat = self.unpatchfier(attn_o)
        
        return spatial_feat + input # residual connection

class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        dropout=0,
    ): 
        super().__init__()

        self.fc1 = DWSConv(in_channels=embed_dim, out_channels=ffn_dim, kernel_size=3)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels=ffn_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0, dilation=1)
        
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        input = x
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x + input

class VSSMBlock(nn.Module):
    def __init__(self, input_size, d_model, dws_kernel, state_dim, ssm_exp_ratio, vssm_drop,
                 patch_size, attn_drop, num_heads,  use_patch_proj, ffn_dim, ffn_dropout):
        super().__init__()
        self.pre_ln1 = LayerNorm2d(d_model)
        self.vssm_mixer = VSSMMixer(input_size=input_size, d_model=d_model, dws_kernel=dws_kernel, state_dim=state_dim, ssm_exp_ratio=ssm_exp_ratio,
                              vssm_drop=vssm_drop, patch_size=patch_size, attn_drop=attn_drop, num_heads=num_heads, use_patch_proj=use_patch_proj)
        self.pre_ln2 = LayerNorm2d(d_model)
        self.ffn = FFN(embed_dim=d_model, ffn_dim=ffn_dim, dropout=ffn_dropout)

    def forward(self, x):
        x = self.pre_ln1(x)
        x = self.vssm_mixer(x)
        x = self.pre_ln2(x)
        x = self.ffn(x)
        return x
        

class VSSMEncoder(nn.Module):
    
    def __init__(self, input_size, d_model, dws_kernel, state_dim, ssm_exp_ratio, vssm_drop, 
                 patch_size, attn_drop, num_heads, use_patch_proj, ffn_dim, ffn_dropout, num_layers):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.encoder.append(VSSMBlock(input_size=input_size, d_model=d_model, dws_kernel=dws_kernel, state_dim=state_dim,
                                          ssm_exp_ratio=ssm_exp_ratio, vssm_drop=vssm_drop, patch_size=patch_size,
                                          attn_drop=attn_drop, num_heads=num_heads, use_patch_proj=use_patch_proj, ffn_dim=ffn_dim, ffn_dropout=ffn_dropout))
            
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.encoder[i](x)
        
        return x
        
        
        
        
        
