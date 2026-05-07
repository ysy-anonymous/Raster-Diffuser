import torch
import torch.nn as nn
from core.diffuser.networks.utils.helpers import ChannelFirstLayerNorm

# Basic Conv1D Block
class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, activation, norm_layer, n_groups=8):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.activation = activation()
        if norm_layer==ChannelFirstLayerNorm or norm_layer==nn.BatchNorm1d:
            self.norm_layer = norm_layer(out_channels)
        elif norm_layer==nn.GroupNorm:
            self.norm_layer = norm_layer(n_groups, out_channels)
        else:
            raise Exception("Normalization must one of ChannelFirstLayerNorm, nn.GroupNorm, nn.BatchNorm1d")
    
    def forward(self, x):
        x = self.conv1d(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x

# Basic ConvTransposed1D Block
class Conv1DTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, activation, norm_layer, n_groups=8):
        super().__init__()
        self.conv1d_trans = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                                            dilation=dilation, groups=groups)
        self.activation = activation()
        if norm_layer==ChannelFirstLayerNorm or norm_layer==nn.BatchNorm1d:
            self.norm_layer = norm_layer(out_channels)
        elif norm_layer==nn.GroupNorm:
            self.norm_layer = norm_layer(n_groups, out_channels)
        else:
            raise Exception("Normalization must one of ChannelFirstLayerNorm, nn.GroupNorm, nn.BatchNorm1d")
    
    def forward(self, x):
        x = self.conv1d_trans(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x

# Basic 2-stack conv1d block with residual connection and film conditions
class Conv1DResBlock(nn.Module):
    """
    2-layer residual convolution block with norm, activations
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 kernel_size, stride, groups, padding, activation, norm_layer, cond_dim, n_groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels != self.out_channels:
            self.residual_connector = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
        self.act_1 = activation()
        if norm_layer==ChannelFirstLayerNorm or norm_layer==nn.BatchNorm1d:
            self.norm_layer = norm_layer(hidden_channels)
            self.norm_layer_2 = norm_layer(out_channels)
        elif norm_layer==nn.GroupNorm:
            self.norm_layer = norm_layer(n_groups, hidden_channels)
            self.norm_layer_2 = norm_layer(n_groups, out_channels)
        else:
            raise Exception("Normalization must one of ChannelFirstLayerNorm, nn.GroupNorm, nn.BatchNorm1d")
        
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
        self.act_2 = activation()

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), activation(), 
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1))
            )
        

    def forward(self, x, cond):
        input = x
        film_feat = self.cond_encoder(cond) # (B, 2*C, 1)
        film_scale, film_shift = film_feat.chunk(2, dim=1) # (B, C, 1)
        film_scale = film_scale + 1.0 # initialized near identical.
        
        x = self.conv1d(x)
        x = self.act_1(x)
        x = self.norm_layer(x)
        
        x = self.conv1d_2(x)
        x = x * film_scale + film_shift # film
        x = self.act_2(x) # selective feature encoding
        x = self.norm_layer_2(x)
        if self.in_channels == self.out_channels:
            return x + input
        else:
            return x + self.residual_connector(input)


class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding, groups, activation, norm_layer, cond_dim, n_groups):
        super().__init__()
        self.downsampler = Conv1DBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=1, dilation=1, activation=activation, norm_layer=norm_layer, n_groups=n_groups)
        self.modulator = Conv1DResBlock(in_channels=out_channels, hidden_channels=out_channels*2, out_channels=out_channels,
                                        kernel_size=3, stride=1, groups=groups, padding="same", activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups)
        self.modulator2 = Conv1DResBlock(in_channels=out_channels, hidden_channels=out_channels*2, out_channels=out_channels,
                                        kernel_size=5, stride=1, groups=groups, padding='same', activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups)
    def forward(self, x, cond):
        x = self.downsampler(x)
        x = self.modulator(x, cond)
        x = self.modulator2(x, cond)
        return x

class UpSampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 padding, out_padding, groups, activation, norm_layer, cond_dim, n_groups):
        super().__init__()
        self.upsampler = Conv1DTransBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, output_padding=out_padding, 
                                            groups=1, dilation=1, activation=activation, norm_layer=norm_layer, n_groups=n_groups)
        self.modulator = Conv1DResBlock(in_channels=out_channels, hidden_channels=out_channels * 2, out_channels=out_channels,
                                        kernel_size=3, stride=1, groups=groups, padding="same", activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups)
        self.modulator2 = Conv1DResBlock(in_channels=out_channels, hidden_channels=out_channels*2, out_channels=out_channels,
                                         kernel_size=5, stride=1, groups=groups, padding='same', activation=activation, norm_layer=norm_layer, cond_dim=cond_dim, n_groups=n_groups)
    def forward(self, x, cond):
        x = self.upsampler(x)
        x = self.modulator(x, cond)
        x = self.modulator2(x, cond)
        return x