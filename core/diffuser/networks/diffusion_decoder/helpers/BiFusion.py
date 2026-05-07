import torch
import torch.nn as nn
import torch.nn.functional as F

# differentiable "soft rasterization" (Gaussian splat)
# This creates a heatmap of where the trajectory currently is.
def rasterize_gaussian(points_xy, H, W, sigma=1.0, coords_are_pixels=True):
    """
    points_xy: (B, T, 2)
      - if coords_are_pixel=True: x in [0..W-1], y in [0..H-1]
      - else: expects normalized grid coords in [-1, 1] and converts to pixels
    returns heat: (B, 1, H, W)
    """
    B, T, _ = points_xy.shape
    device = points_xy.device
    dtype = points_xy.dtype

    if not coords_are_pixels:
        # points in [-1, 1] -> pixel coords
        x = (points_xy[..., 0] + 1) * 0.5 * (W-1)
        y = (points_xy[..., 1] + 1) * 0.5 * (H-1)
        pts = torch.stack([x, y], dim=-1)
    else:
        pts = points_xy
    
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')      # (H, W)
    grid = torch.stack([xx, yy], dim=-1)[None, None]    # (1, 1, H, W, 2)

    # (B, T, 1, 1, 2) - (1, 1, H, W, 2) -> (B, T, H, W, 2)
    diff = pts[:, :, None, None, :] - grid
    d2 = (diff ** 2).sum(dim=-1) # (B, T, H, W)

    heat = torch.exp(-0.5 * d2 / (sigma ** 2 + 1e-8)) # (B, T, H, W)

    # Combine across time. "max" makes a clean path footprint.
    heat = heat.max(dim=1).values.unsqueeze(1) # (B, 1, H, W)
    return heat

# rasterization with velocity map
# output 3 channels feature map
def rasterize_with_velocity(points_xy, H, W, sigma=1.0, coords_are_pixels=True):
    heat = rasterize_gaussian(points_xy, H, W, sigma, coords_are_pixels) # (B, 1, H, W)

    v = points_xy[:, 1:] - points_xy[:, :-1]
    v = F.pad(v, (0, 0, 1, 0)) # (B, T, 2) align with points

    # weight velocities by per-time gaussian splats
    # re-create per-time heat to do weighted average (still cheap at 32x32)
    B, T, _ = points_xy.shape
    device, dtype = points_xy.device, points_xy.dtype
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)[None, None] # (1, 1, H, W, 2)

    pts = points_xy if coords_are_pixels else torch.stack([
        (points_xy[..., 0] + 1) * 0.5 * (W-1),
        (points_xy[..., 1] + 1) * 0.5 * (H-1)
    ], dim=-1)

    diff = pts[:, :, None, None, :] - grid
    d2 = (diff ** 2).sum(dim=-1)
    w = torch.exp(-0.5 * d2 / (sigma ** 2 + 1e-8)) # (B, T, H, W)

    vx = (w * v[:, :, None, None, 0]).sum(dim=1)
    vy = (w * v[:, :, None, None, 1]).sum(dim=1)
    wsum = w.sum(dim=1).clamp_min(1e-6)

    vx = (vx / wsum).unsqueeze(1)
    vy = (vy / wsum).unsqueeze(1)

    return torch.cat([heat, vx, vy], dim=1) # (B, 3, H, W)


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
    def __init__(self, cm, cp, cs, ct, n_blocks=3):
        super().__init__()
        self.in_proj = nn.Conv2d(cm + cp + ct, cs, 1)
        self.blocks = nn.Sequential(*[ResBlock2D(cs) for _ in range(n_blocks)])
        self.out = nn.Conv2d(cs, cs, 1)
    
    def forward(self, M, P, t_embed):
        # M: (B, Cm, H, W), P: (B, Cp, H, W), t_embed: (B, Ct)
        B, _, H, W = M.shape
        t = t_embed[:, :, None, None].expand(B, t_embed.size(1), H, W)
        x = torch.cat([M, P, t], dim=1)
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

class BiDirectionalFusionBlock(nn.Module):
    def __init__(self, cm, cp, cs, ct, Hs=32, Ws=32, R=3, sigma=1.2, update_coeff=0.2, num_spatial_blocks=3, num_correct_dim=128, weight_sharing=True):
        super().__init__()
        self.Hs, self.Ws = Hs, Ws
        self.cp = cp
        self.R = R
        self.sigma = sigma
        self.weight_sharing = weight_sharing

        if isinstance(update_coeff, (int, float)):
            self.learnable_update_coeff = False
            self._update_coeff = update_coeff
        elif isinstance(update_coeff, str) and update_coeff == 'learnable':
            self.learnable_update_coeff = True
            self.max_update_coeff = 1.0
            init_coeff = 0.05 # start with a small update coeff for stability, can be tuned or ablated
            init_ratio = init_coeff / self.max_update_coeff
            init_ratio = min(max(init_ratio, 1e-4), 1.0-1e-4) # clip to [0.1, init_ratio] to prevent too small values 

            init_logit = torch.logit(torch.tensor(init_ratio))
            self.update_coeff_logit = nn.Parameter(init_logit.clone())
        else:
            raise ValueError(f"update_coeff must be a float/int or 'learnable', got {update_coeff}")

        # weight sharing for spatial refine blocks across iterations.
        if self.weight_sharing:
            self.spatial = SpatialRefine(cm=cm, cp=cp, cs=cs, ct=ct, n_blocks=num_spatial_blocks)
            self.correct = PointCorrector(cs=cs, ct=ct, hidden=num_correct_dim)
        else:
            self.spatial_blocks = nn.ModuleList([
                SpatialRefine(cm=cm, cp=cp, cs=cs, ct=ct, n_blocks=num_spatial_blocks) for _ in range(R)
            ])
            self.correct_blocks = nn.ModuleList([
                PointCorrector(cs=cs, ct=ct, hidden=num_correct_dim) for _ in range(R)
            ])

    @property
    def update_coeff(self):
        if self.learnable_update_coeff:
            return self.max_update_coeff * torch.sigmoid(self.update_coeff_logit)
        
        return self._update_coeff


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
        for i in range(self.R):
            # WRITE (canvas in pixel coords)
            # convert normalized [-1, 1] -> pixels for rasterizer
            x_pix = torch.stack([
                (x[..., 0] + 1) * 0.5 * (self.Ws - 1),
                (x[..., 1] + 1) * 0.5 * (self.Hs - 1),
            ], dim=-1)

            if self.cp == 1:
                P = rasterize_gaussian(x_pix, self.Hs, self.Ws, sigma=self.sigma, coords_are_pixels=True) # (B, 1, Hs, Ws)
            else: # cp must be 3 in this case
                P = rasterize_with_velocity(x_pix, self.Hs, self.Ws, sigma=self.sigma, coords_are_pixels=True) # (B, 3, Hs, Ws)

            if self.weight_sharing:
                # SPATIAL REFINE
                S = self.spatial(M, P.to(M.dtype), t_embed) # (B, Cs, Hs, Ws)

                # READ
                read = sample_spatial(S, x) # (B, T, Cs)

                # CORRECT (small step for stability)
                dx = self.correct(read, t_embed) # (B, T, 2)
            else:
                S = self.spatial_blocks[i](M, P.to(M.dtype), t_embed) # (B, Cs, Hs, Ws)
                read = sample_spatial(S, x) # (B, T, Cs)
                dx = self.correct_blocks[i](read, t_embed) # (B, T< 2)


            x = x + self.update_coeff * dx

            # optional: keep in [-1, 1]
            x = x.clamp(-1, 1)

        return x