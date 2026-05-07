import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.LayerNorm):
    def __init__(self, channels):
        super().__init__(normalized_shape=channels, eps=1e-6)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()
    
class GroupNorm(nn.GroupNorm):
    def __init__(self, channels):
        super().__init__(num_groups=1, num_channels=channels, eps=1e-6)
    
    def forward(self, x):
        x = super().forward(x)
        return x

# Channel First layer normalization
class ChannelFirstLayerNorm(nn.Module):
    """
        Layer Normalization over channel first version. (B, C, L) instead of (B, L, C)
    """
    def __init__(self, num_feats, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_feats))
            self.bias = nn.Parameter(torch.zeros(num_feats))

    def forward(self, x): # x: (N, C, L)
        # mean/variance over channels at each (N, L)
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True) # (N, 1, L)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x
        

# patchify function
class Patchfier(nn.Module):
    def __init__(self, input_size: tuple, patch_size: tuple, style: int):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size # given in (H, W)
        self.style = style
        if input_size != None:
            if input_size[0] % patch_size[0] != 0 or input_size[1] % patch_size[1] != 0:
                raise Exception("Input height, width must be divisible by patch size")
            if not style in [0, 1]:
                raise Exception("Select one of followings: [0: vit style token, 1: conv1d style token]")
        self.unfolder = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.unfolder(x) # (B, c * ph * pw, L)
        if self.style == 0:
            x = x.permute(0, 2, 1) # (B, L, c * ph * pw)
        return x
    
# reverse process of patchify function
class UnPatchfier(nn.Module):
    def __init__(self, input_size: tuple, patch_size: tuple, feat_channels: int):
        super().__init__()
        self.input_size = input_size # (H, W)
        self.patch_size = patch_size # (PH, PW)
        self.feat_channels = feat_channels # (C)
        
        self.H, self.W = input_size
        self.PH, self.PW = patch_size
        
        self.NH, self.NW = self.H//self.PH, self.W//self.PW
        self.N = self.NH * self.NW
        if self.H % self.PH !=0 or self.W % self.PW != 0:
            raise Exception("Restore Target H, W Must be devisible by patch size")
        
    def forward(self, x):
        # x: (B, L, D)
        B, S, D = x.shape
        
        if D != self.feat_channels * self.patch_size[0] * self.patch_size[1]:
            raise Exception(f"Dimension of input must be equal to feat_channels * patch_size[0] * patch_size[1]={self.feat_channels * self.PH * self.PW} for exact unpatchify")
        if S != self.N:
            raise Exception(f"Sequence Length({S}) of input must be equal to total number of patches! ({self.N})")
        
        x = x.view(B, self.NH, self.NW, self.feat_channels, self.PH, self.PW) # (B, NH, NW, C, PH, PW)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous() # (B, C, NH, PH, NW, PW)
        return x.view(B, self.feat_channels, self.H, self.W) # (B, C, NH * PH, NW * PW)
    

# Used to upsample map feature to (32, 32)
class FeatUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_ratio : float, upsample_method : int):
        super().__init__()
        self.in_channels = in_channels # number of input channels
        self.out_channels = out_channels # number of output channels
        self.upsample_ratio = upsample_ratio # upsampling ratio
        self.upsample_method = upsample_method
        
        if self.upsample_method > 1.0:
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
        else:
            self.identity = torch.nn.Identity()

    def forward(self, x):
        # if upsample ratio equals to 1.0, skip upsampling
        if self.upsample_ratio == 1.0:
            return self.identity(x)
        else:
            if self.upsample_method in [0, 1]:
                x = self.upsampler(x)
            elif self.upsample_method == 2:
                x = F.interpolate(x, scale_factor=self.upsample_ratio)
                x = self.upsampler(x)
                        
        return x
    
