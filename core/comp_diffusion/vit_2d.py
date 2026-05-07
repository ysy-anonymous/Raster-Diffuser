# Vit + Convolution Hybrid Implementation
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.diffuser.networks.utils.helpers import UnPatchfier, Patchfier
from core.diffuser.networks.utils.enums import PatchifyStyle


# helpers
# From https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# Convolutional Block
class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        super().__init__()
        self.conv_2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv_2d(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# Residual Conv Block with Squeeze-Excitation Network.
class ConvResBlockSE(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, groups):
        super().__init__()
        self.res_1 = Conv2DBlock(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups)
        self.res_2 = Conv2DBlock(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=1)
        self.res_3 = Conv2DBlock(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=1)
        
        r = 4
        self.squeeze_ext = nn.Sequential(nn.Linear(in_features=in_channels, out_features= in_channels//r, bias=True),
                                             nn.ReLU(),
                                             nn.Linear(in_features=in_channels//r, out_features=in_channels, bias=True),
                                             nn.Sigmoid())
    
    def forward(self, x):
        input_x = x
        pool_x = F.avg_pool2d(x, x.shape[2: ]).flatten(1, -1)
        chan_weights = self.squeeze_ext(pool_x) # (B, C)
        
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        
        x = x * chan_weights[:, :, None, None] # channel-wise broadcasted
        x = x + input_x # residual connection
        return x
        

# Convert ViT Encoded Feature to Convolution
class ViT2ConvBridge(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super().__init__()
        self.conv_se = ConvResBlockSE(in_channels=in_channels, hidden_channels=in_channels*4, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size//2, dilation=1, groups=groups)
        self.conv_se2 = ConvResBlockSE(in_channels=in_channels, hidden_channels=in_channels*4, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size//2, dilation=1, groups=groups)
        self.out_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                  dilation=1, groups=1)
        
    def forward(self, x):
        x = self.conv_se(x)
        x = self.conv_se2(x)
        x_out = self.out_conv(x)
        return x_out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, stem_channels, cond_out_dim, featmap_out_dim, dim, depth, heads, mlp_dim, pool = 'cls', 
                 in_channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        pool = pool.lower()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.stem_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels, kernel_size=3, stride=1, padding='same'),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=stem_channels, out_channels=stem_channels, kernel_size=3, stride=1, padding='same'),
                                        nn.ReLU())
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = stem_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Patchfier(input_size=pair(image_size), patch_size=pair(patch_size), style=PatchifyStyle.VIT_STYLE.value), # 1. Patchfy
            nn.Linear(patch_dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )
        
        dim_after_unpatch = dim // (patch_height * patch_width)
        self.patch_to_featmap = UnPatchfier(input_size=pair(image_size), patch_size=pair(patch_size), feat_channels=dim_after_unpatch)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        
        self.vit2conv_bridge = ViT2ConvBridge(in_channels=dim_after_unpatch, out_channels=featmap_out_dim, kernel_size=3, groups=1)
        self.mlp_head = nn.Linear(dim+featmap_out_dim, cond_out_dim)
        

    def forward(self, img):
        # stem convolution
        x = self.stem_conv(img)

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # print("x.shape: ", x.shape)
        # print("pos_embedding shape: ", self.pos_embedding[:, :(n+1)].shape)
        
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.pool == 'mean':
            pool_x = x.mean(dim=1)
            spatial_map = self.patch_to_featmap(x)
        elif self.pool == 'cls':
            pool_x = x[:, 0]
            spatial_map = self.patch_to_featmap(x[:, 1:])
        spatial_x = self.vit2conv_bridge(spatial_map)

        # enhance map condition token by spatial conv features
        pool_x = torch.cat([pool_x, F.avg_pool2d(spatial_x, kernel_size=spatial_x.shape[2:]).flatten(start_dim=1, end_dim=-1)], dim=-1)
        pool_x = self.mlp_head(pool_x)
        
        return spatial_x, pool_x
            
        