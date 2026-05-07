import torch
import torch.nn.functional as F

def collision_cost(obs_mask, traj_norm):
    """
    obs_mask: (B,1,H,W) in {0,1}
    traj_norm: (B,K,T,2) in [-1,1]
    returns cost: (B,K)
    """
    B, K, T, _ = traj_norm.shape
    H, W = obs_mask.shape[-2:]
    
    # sample occupancy at points
    grid = traj_norm.view(B*K, T, 1, 2) # (BK, T, 1, 2)
    mask_rep = obs_mask[:, None].expand(B, K, 1, H, W).reshape(B*K, 1, H, W)
    occ = F.grid_sample(mask_rep, grid, align_corners=False, mode='bilinear', padding_mode='zeros') # [B*K, 1, T, 1]
    occ = occ.squeeze(1).squeeze(-1) # (BK, T)
    
    # also sample along segments (midpoints) to catch "cut through obstacles"
    mids = 0.5 * (traj_norm[:, :, 1:] + traj_norm[:, :, :-1])
    grid_m = mids.view(B*K, T-1, 1, 2)
    occ_m = F.grid_sample(mask_rep, grid_m, align_corners=False, mode='bilinear', padding_mode='zeros') # [B*K, 1, T-1, 1]
    occ_m = occ_m.squeeze(1).squeeze(-1)
    
    cost = occ.mean(dim=1) + occ_m.mean(dim=1) # (BK, )
    return cost.view(B, K)