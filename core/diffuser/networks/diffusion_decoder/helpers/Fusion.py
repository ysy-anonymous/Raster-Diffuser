import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Blocks that consist of 2 stacked convolution layer
class ResBlock2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels)
        )
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(x + self.net(x))


class SpatialRefine(nn.Module):
    def __init__(self, cm, cs, ct, n_blocks=3):
        super().__init__()
        self.in_proj = nn.Conv2d(cm + ct, cs, 1)
        self.blocks = nn.Sequential(*[ResBlock2D(cs) for _ in range(n_blocks)])
        self.out = nn.Conv2d(cs, cs, 1)
    
    def forward(self, M, t_embed):
        # M: (B, Cm, H, W), t_embed: (B, Ct)
        B, _, H, W = M.shape
        t = t_embed[:, :, None, None].expand(B, t_embed.size(1), H, W)
        x = torch.cat([M, t], dim=1)
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out(x)

def sample_spatial(S, points_xy_norm):
    """
    S: (B, C, H, W)
    points_xy_norm: (B, T, 2) in [-1, 1] for grid_sample
    returns: (B, T, C)
    """
    B, C, H, W = S.shape
    grid = points_xy_norm.view(B, -1, 1, 2) # (B, T, 1, 2)
    feat = F.grid_sample(S, grid, mode='bilinear', align_corners=False)
    return feat.squeeze(-1).transpose(1, 2) # (B, T, C)


class PointCorrector(nn.Module):
    def __init__(self, cs, ct, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cs + ct, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )
    
    def forward(self, read_feat, t_embed):
        # read_feat: (B, T, Cs), t_embed: (B, Ct)
        B, T, Cs = read_feat.shape
        t = t_embed[:, None, :].expand(B, T, t_embed.size(1))
        dx = self.mlp(torch.cat([read_feat, t], dim=-1))
        return dx
        

# Simple fusion block that performs R rounds of refinement using spatial map information.
class SimpleFusionBlock(nn.Module):
    def __init__(self, cm, cs, ct, Hs=32, Ws=32, R=3):
        super().__init__()
        self.Hs, self.Ws = Hs, Ws
        self.R = R

        self.spatial = SpatialRefine(cm=cm, cs=cs, ct=ct, n_blocks=3)
        self.correct = PointCorrector(cs=cs, ct=ct, hidden=128)
    
    def forward(self, M, x0_hat_norm, t_embed):
        """
        M: (B, Cm, H, W)
        x0_hat_norm: (B, T, 2) normalized to [-1, 1] in the *same BEV frame* as map
        t_embed: (B, Ct)
        returns refined x0_hat_norm
        """
        map_in = M
        # Upsample map feature to working resolution
        M = F.interpolate(map_in, size=(self.Hs, self.Ws), mode='bilinear', align_corners=False)

        x = x0_hat_norm
        for _ in range(self.R):
            # SPATIAL REFINE
            S = self.spatial(M, t_embed) # (B, Cs, Hs, Ws)

            # READ
            read = sample_spatial(S, x) # (B, T, Cs)

            # CORRECT (small step for stability)
            dx = self.correct(read, t_embed) # (B, T, 2)
            x = x + 0.2 * dx 

            # optional: keep in [-1, 1]
            x = x.clamp(-1, 1)

        return x