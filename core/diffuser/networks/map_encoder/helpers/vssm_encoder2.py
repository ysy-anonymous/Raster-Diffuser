import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

# import VSSM
from core.diffuser.networks.map_encoder.helpers.segman_encoder import VSSM, LayerNorm2d

# import DropPath from timm library
from timm.layers.drop import DropPath

# Depth-wise Convolution
class DWConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True, groups=in_channels)
    
    def forward(self, x):
        x = self.dw_conv(x)
        return x

# Depth-wise Seperable Convolution
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

# Borrow Code from segman_encoder.py, https://github.com/yunxiangfu2001/SegMAN
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()
        
        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias
          
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
    
    def extra_repr(self) -> str:
        return '{dim}, init_value={init_value}, bias={enable_bias}'.format(**self.__dict__)

# VSSM Based mixer
# 2 Stack of VSSM layer with depth-wise convolution
class VSSMMixer(nn.Module):
    def __init__(self, d_model, dws_kernel=3, state_dim=1, ssm_exp_ratio=1, vssm_drop=0.0):
        super().__init__()
        
        # Convolution-SSM Token Mixer
        self.conv_ssm_mixer = nn.Sequential(
            DWConv(in_channels=d_model, kernel_size=3),
            VSSM(d_model=d_model, d_state=state_dim, expansion_ratio=ssm_exp_ratio, dropout=vssm_drop) # VSSM Layer
        )
        
        # DWS Convolution (mini-token-channels mixer)
        self.dws_conv = nn.Sequential(
            DWSConv(in_channels=d_model, out_channels=d_model * 2, kernel_size=dws_kernel),
            nn.GELU(),
            DWSConv(in_channels=d_model * 2, out_channels= d_model, kernel_size=dws_kernel),
        )

        # Convolution-SSM Token Mixer
        self.conv_ssm_mixer_2 = nn.Sequential(
            DWConv(in_channels=d_model, kernel_size=3),
            VSSM(d_model=d_model, d_state=state_dim, expansion_ratio=ssm_exp_ratio, dropout=vssm_drop)
        )
        
    def forward(self, x):
        """
        x: (B, C, H, W) 2D spatial tensor
        """
        input_x = x
        B, C, H, W = x.shape
        x = self.conv_ssm_mixer(x)
        x = x.reshape(B, C, H, W).contiguous()
        x = self.dws_conv(x)
        x = self.conv_ssm_mixer_2(x)
        x = x.reshape(B, C, H, W).contiguous()

        return x + input_x # residual connection


# Borrow Code from segman_encoder.py, https://github.com/yunxiangfu2001/SegMAN
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0,
    ): 
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x + input # residual connection


class VSSMBlock(nn.Module):
    def __init__(self, d_model, dws_kernel, state_dim, ssm_exp_ratio, vssm_drop, ffn_dim, ffn_drop):
        super().__init__()
        self.pre_ln1 = LayerNorm2d(d_model)
        self.vssm_mixer = VSSMMixer(d_model=d_model, dws_kernel=dws_kernel, state_dim=state_dim, ssm_exp_ratio=ssm_exp_ratio, vssm_drop=vssm_drop)
        self.pre_ln2 = LayerNorm2d(d_model)
        self.ffn = FFN(embed_dim=d_model, ffn_dim=ffn_dim, dropout=ffn_drop)

    def forward(self, x):
        x = self.pre_ln1(x)
        x = self.vssm_mixer(x)
        x = self.pre_ln2(x)
        x = self.ffn(x)
        return x
        

class VSSMEncoder(nn.Module):
    def __init__(self, d_model, dws_kernel, state_dim, ssm_exp_ratio, vssm_drop, ffn_dim, ffn_drop, num_layers):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.encoder.append(VSSMBlock(d_model=d_model, dws_kernel=dws_kernel, state_dim=state_dim, ssm_exp_ratio=ssm_exp_ratio, vssm_drop=vssm_drop, ffn_dim=ffn_dim, ffn_drop=ffn_drop))
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.encoder[i](x)
        
        return x
        
        
        
        
        
