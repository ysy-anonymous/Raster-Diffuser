import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from core.diffuser.networks.utils.helpers import LayerNorm, GroupNorm



class TransposedConvBlock(nn.Module):
    """
    Basic Transposed Conv - Norm - Act layer block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, padding, out_padding, activation, norm_layer):
        super().__init__()
        self.transpose_conv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                                   kernel_size = kernel_size, stride=stride, padding=padding, output_padding=out_padding, groups=groups, 
                                                   bias=False)
        self.act = activation()
        if norm_layer==LayerNorm or norm_layer==GroupNorm or norm_layer==nn.BatchNorm2d:
            self.norm_layer = norm_layer(out_channels)
        else:
            raise Exception("Normalization must one of LayerNorm, GroupNorm, nn.BatchNorm2d")
        
    def forward(self, x):
        x = self.transpose_conv2d(x)
        x = self.norm_layer(x)
        x = self.act(x)
        
        return x


class ConvBlock(nn.Module):
    """
    Basic Conv - Norm - Act layer block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, padding, activation, norm_layer):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                groups=groups, padding=padding, bias=False)
        self.act = activation()
        if norm_layer==LayerNorm or norm_layer==GroupNorm or norm_layer==nn.BatchNorm2d:
            self.norm_layer = norm_layer(out_channels)
        else:
            raise Exception("Normalization must be one of LayerNorm, GroupNorm, nn.BatchNorm2d")
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm_layer(x)
        x = self.act(x)
        
        return x


class ConvResBlock(nn.Module):
    """
    2-layer residual convolution block with norm, activations
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 kernel_size, stride, groups, padding, activation, norm_layer):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding
                
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                              stride=stride, bias=False, groups=groups, padding=padding)
        if norm_layer==LayerNorm or norm_layer==GroupNorm or norm_layer==nn.BatchNorm2d:
            self.norm = norm_layer(hidden_channels)
        else:
            raise Exception("Normalization must one of LayerNorm, GroupNorm, nn.BatchNorm2d")
        self.act1 = activation()
        
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, bias=False, groups=groups, padding=padding)
        if norm_layer==LayerNorm or norm_layer==GroupNorm or norm_layer==nn.BatchNorm2d:
            self.norm2 = norm_layer(out_channels)
        else:
            raise Exception("Normalization must one of LayerNorm, GroupNorm, nn.BatchNorm2d")
        self.act2 = activation()

        self.res_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True)
        
    def forward(self, x):
        input = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        if self.in_channels != self.out_channels:
            return x + self.res_conv(input) # residual connection via 1x1 projection
        else:
            return x + input # residual connection

class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, padding, activation, norm_layer, modulator_kernel=2):
        super().__init__()
        self.down_samples = ConvBlock(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, groups=groups, padding=padding,
                                      activation=activation, norm_layer=norm_layer)
        self.modulator = ConvResBlock(in_channels=out_channels, hidden_channels=out_channels * 2, out_channels=out_channels,
                                      kernel_size=modulator_kernel, stride=1, groups=1, padding='same', activation=activation, norm_layer=norm_layer)
    
    def forward(self, x):
        x = self.down_samples(x)
        x = self.modulator(x)
        return x
        
class UpSampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, padding, out_padding, activation, norm_layer, modulator_kernel=2):
        super().__init__()
        self.up_samples = TransposedConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                              stride=stride, groups=groups, padding=padding, out_padding=out_padding, 
                                              activation=activation, norm_layer=norm_layer)
        self.modulator = ConvResBlock(in_channels=out_channels, hidden_channels=out_channels * 2, out_channels=out_channels,
                                      kernel_size=modulator_kernel, stride=1, groups=1, padding='same', activation=activation, norm_layer=norm_layer)
        
    def forward(self, x):
        x = self.up_samples(x)
        x = self.modulator(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_scheme, stride_scheme, latent_kernel, activation, norm_layer, modulator_scheme):
        super().__init__()
        self.in_channel = in_channel # input channel (start channel of unet)
        self.out_channel = out_channel # output channel from unet
        self.kernel_scheme = kernel_scheme
        self.stride_scheme = stride_scheme
        self.latent_kernel = latent_kernel # latent kernel size
        self.activation = activation
        self.norm_layer = norm_layer
        self.modulator_scheme = modulator_scheme
        modulator_reverse = list(reversed(modulator_scheme))
        
        self.down_sampler = nn.ModuleList([])
        self.up_sampler = nn.ModuleList([])
        
        start_channel = in_channel
        all_channels = [start_channel] + [start_channel * (2**n) for n in range(1, len(kernel_scheme)+1)]
        last_channel = all_channels[-1]
        for i, (kernel_sz, stride) in enumerate(zip(kernel_scheme, stride_scheme), 0):
            self.down_sampler.append(DownSampler(in_channels=start_channel, out_channels=start_channel*2, kernel_size=kernel_sz, stride=stride,
                                                 groups=1, padding=0, activation=self.activation, norm_layer=self.norm_layer, modulator_kernel=modulator_scheme[i]))
            start_channel = start_channel * 2
        
        self.latent_conv = ConvBlock(in_channels=all_channels[-1], out_channels=all_channels[-1], kernel_size=latent_kernel, stride=1,
                                     groups=1, padding=0, activation=activation, norm_layer=norm_layer)
        
        for i, (kernel_sz, stride) in enumerate(reversed(list(zip(kernel_scheme, stride_scheme))), 0):
            self.up_sampler.append(UpSampler(in_channels=last_channel, out_channels=last_channel//2, kernel_size=kernel_sz, stride=stride, 
                                             groups=1, padding=0, out_padding=0, activation=self.activation, norm_layer=self.norm_layer, modulator_kernel=modulator_reverse[i]))
            last_channel = last_channel // 2 + all_channels[-2 - i]
            
        self.modulator = nn.Conv2d(in_channels=last_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, 
                                   dilation=1, groups=1, bias=True)
        
    def forward(self, x):
        downsampled_feats = []
        for i in range(len(self.down_sampler)):
            downsampled_feats.append(x)
            x = self.down_sampler[i](x)
        
        x = self.latent_conv(x)
            
        last_idx = len(downsampled_feats)-1
        for i in range(len(self.up_sampler)):
            x = self.up_sampler[i](x)
            x = torch.cat([x, downsampled_feats[last_idx-i]], dim=1)
        x = self.modulator(x)
        return x


class MapEncoder(nn.Module):
    """
    Convolution based map encoder.
    This extracts the binary map (single-channel) by non-overlapping convolution
    and upsample the down-sampled feature map again to match the original shape.
    
    Also the network uses patch-unshuffle algorithm to reshape nearby pixel
    to the channel direction (lossless compression) and reshape the feature map to original shape
    """
    
    def __init__(self, map_size: tuple, stem_in: int, stem_out: int, unet_out: List, map_out: int, kernel_schemes: List[List], stride_schemes: List[List], modulator_schemes: List[List],
                 encoder_act: nn.Module, encoder_norm: nn.Module):
        super().__init__()
        self.kernel_schemes = kernel_schemes
        self.stride_schemes = stride_schemes
        self.modulator_schemes = modulator_schemes
        
        self.encoder_act = encoder_act
        self.encoder_norm = encoder_norm
        
        # basic stem layer that uses standard residual convolution block
        self.stem_layers = nn.Sequential(
            ConvResBlock(in_channels=stem_in+2, hidden_channels=16, out_channels=32, kernel_size=3,
                              stride=1, groups=1, padding='same', activation=self.encoder_act, norm_layer=self.encoder_norm),
            ConvResBlock(in_channels=32, hidden_channels=64, out_channels=stem_out, kernel_size=3,
                         stride=1, groups=1, padding='same', activation=self.encoder_act, norm_layer=self.encoder_norm)
        )
        
        # build multiple-unet
        self.unet_family = nn.ModuleList([])
        for i, (kernel_scheme, stride_scheme, modulator_scheme) in enumerate(zip(kernel_schemes, stride_schemes, modulator_schemes), 0):
            sub_unet = UNet(in_channel=stem_out, out_channel=unet_out[i], kernel_scheme=kernel_scheme, stride_scheme=stride_scheme,
                            latent_kernel=1, activation=self.encoder_act, norm_layer=self.encoder_norm, modulator_scheme=modulator_scheme)
            self.unet_family.append(sub_unet)
            
        self.modulator = nn.Conv2d(in_channels=sum(unet_out), out_channels=map_out, kernel_size=1, stride=1, padding=0)
        
        # default map size: (8, 8)
        self.H, self.W = map_size
        pos_map = self._create_pos_grid((self.H, self.W)) # (1, 2, grid_H, grid_W)
        self.register_buffer("pos_map", pos_map)
        
        self.init_weight()
    
    # create positional grid map
    def _create_pos_grid(self, size):
        H, W = size
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pos_map = torch.stack([ys, xs], dim=0).unsqueeze(0).float()
        return pos_map
    
    # set positional grid map
    def _set_pos_grid(self, size):
        H, W = size
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pos_map = torch.stack([ys, xs], dim=0).unsqueeze(0).float()
        self.H = H; self.W = W
        self.pos_map = pos_map
    
    # initialize weights
    def init_weight(self):
        def _init(m: nn.Module):
            # ---- Convs / Linears ----
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.Linear)):
                if self.encoder_act in [nn.ReLU, nn.LeakyReLU]:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # ---- Norm layers ----
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, LayerNorm,
                                GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            
            # ---- Embeddings (if any appear inside UNet etc.) ----
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
        
        # Initialize everything inside this modules (stem + all UNets + modulator)
        self.apply(_init)
        
        # Optional: stablize output projection
        # nn.init.zeros_(self.modulator.weight)
        # if self.modulator.bias is not None:
            # nn.init.zeros_(self.modulator.bias)
        
        
    def forward(self, x):
        """
        x: 8x8 binary obstacle map
        """
        
        pos_grid = self.pos_map.expand(x.shape[0], -1, -1, -1)
        pos_grid = pos_grid.to(dtype=x.dtype).to(x.device)
        x = torch.cat([x, pos_grid], dim=1)
        # start stem layers
        stem_feat = self.stem_layers(x)
        
        # multi sub unet
        unet_outs = []
        for i in range(len(self.unet_family)):
            unet_o = self.unet_family[i](stem_feat)
            unet_outs.append(unet_o)
        
        out_features = torch.cat(unet_outs, dim=1) # [B, C, H, W]
        outs = self.modulator(out_features)
        
        return outs

def main():    
    import torchinfo
        
    map_encoder = MapEncoder(stem_out=64, unet_out=[256, 256], map_out=768,
                            kernel_schemes=[[2, 2, 2], [4, 2, 1]], stride_schemes=[[2, 2, 2], [4, 2, 1]], modulator_schemes=[[2, 1, 1], [2, 1, 1]],
                            encoder_act=nn.GELU, encoder_norm=LayerNorm)
    torchinfo.summary(map_encoder, input_size=(32, 1, 8, 8))
    map_encoder = map_encoder.to('cuda')
    map_output = map_encoder(torch.randn(32, 1, 8, 8).to('cuda'))
    
    # print("map encoder output: ", map_output)
    print("map encoder output shape: ", map_output.shape)

if __name__ == "__main__":
    main()