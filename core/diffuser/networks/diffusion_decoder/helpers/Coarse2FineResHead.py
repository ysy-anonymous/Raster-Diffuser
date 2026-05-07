import torch
import torch.nn as nn
import torch.nn.functional as F

def interpolate_time(x, T_out):
    return F.interpolate(x.transpose(1,2), size=T_out, mode='linear', align_corners=True).transpose(1, 2)

def sample_map_along_traj(map_cond, traj_norm, padding_mode='zeros'):
    """
    map_cond: (B, Cm, H, W)
    traj_norm: (B, T, 2) in [-1, 1] with order (x, y) for grid_sample
    returns: (B, T, Cm)
    """
    grid = traj_norm.unsqueeze(2) # (B, T, 1, 2)
    feat = F.grid_sample(
        input=map_cond, grid=grid,
        mode='bilinear', align_corners=False, padding_mode=padding_mode
    ) # (B, Cm, T, 1)
    feat = feat.squeeze(-1).transpose(1,2) # (B, T, Cm)
    return feat


# FiLM based Conditioning Head
class FiLMHead1D(nn.Module):
    def __init__(self, in_dim, cond_dim, hidden_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # FiLM parameters from global cond
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        )
    
    def forward(self, x, global_cond):
        """
        x: (B, T, in_dim)
        global_cond: (B, cond_dim)
        returns: (B, T, out_dim)
        """
        B, T, _ = x.shape
        x = self.norm(x)
        x = self.fc1(x) # (B, T, hidden_dim)

        gamma_beta = self.to_gamma_beta(global_cond) # (B, 2*hidden_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1) # each (B, hidden_dim)
        gamma = gamma.unsqueeze(1) # (B, 1, hidden_dim)
        beta = beta.unsqueeze(1) # (B, 1, hidden_dim)

        x = x * (gamma + 1) + beta
        h = F.silu(x)
        return self.fc2(h) # (B, T, out_dim)



class Coarse2FineRefiner(nn.Module):
    """
    Coarse to fine trajectory refiner based on multi-time resolution interpolated trajectory features.
    This head inputs:
        - in_feat: (B, T_fine, C) trajectory features from the last layer of the diffusion decoder
        - map_cond: (B, Cm, H, W) map condition features from the map encoder
        - global_cond: (B, Cg) global condition features that is concatenated feature of [pooled_map_cond, obs_cond, timestep_cond]
        - x_t_in: (B, T_fine, 2) input noisy trajectory at current diffusion step
        - timesteps: (B, ) integer diffusion timesteps
        - alphas_cumprod: precomputed noise scheduler alphas_cumprod for diffusion step embedding
    """
    def __init__(self, T_coarse, T_mid, T_fine, in_feat_dim, map_dim, global_cond_dim, hidden_dim):
        super().__init__()
        assert T_coarse < T_mid and T_mid < T_fine, "T_coarse < T_mid < T_fine must hold"
        self.T_coarse = T_coarse
        self.T_mid = T_mid
        self.T_fine = T_fine

        # Multi-time resolution trajectory heads using FiLM conditioning
        self.head_coarse = FiLMHead1D(in_dim=in_feat_dim+map_dim, cond_dim=global_cond_dim, hidden_dim=hidden_dim, out_dim=2)
        # self.head_coarse = FiLMHead1D(in_dim=in_feat_dim, cond_dim=global_cond_dim, hidden_dim=hidden_dim, out_dim=2)
        self.head_mid = FiLMHead1D(in_dim=in_feat_dim+map_dim, cond_dim=global_cond_dim, hidden_dim=hidden_dim, out_dim=2)
        self.head_fine = FiLMHead1D(in_dim=in_feat_dim+map_dim, cond_dim=global_cond_dim, hidden_dim=hidden_dim, out_dim=2)

        
    def forward(self, in_feat, map_cond, global_cond, x_t_in, timesteps, alphas_cumprod):
        B, T, Ch = in_feat.shape
        assert T == self.T_fine, "Input dimension of trajectory feature must be same as T_fine"

        # diffusion coefficients for the batch
        a_bar = alphas_cumprod[timesteps].view(B, 1, 1) # (B, 1, 1)
        sqrt_ab = torch.sqrt(a_bar) # (B, 1, 1)
        sqrt_omab = torch.sqrt(1 - a_bar)

        # ---- Stage 0 : Coarse residual at T_coarse (conditioned on sampled traj feature from x_t_in) ----
        init_coarse = interpolate_time(x_t_in, self.T_coarse) # (B, T_coarse, 2)
        h_coarse = interpolate_time(in_feat, self.T_coarse) # (B, T_coarse, Ch)
        map_sampled_coarse = sample_map_along_traj(map_cond, init_coarse) # (B, T_coarse, map_dim)

        traj_coarse = self.head_coarse(torch.cat([h_coarse, map_sampled_coarse], dim=-1), global_cond) # (B, T_coarse, 2)
        # traj_coarse = self.head_coarse(h_coarse, global_cond) # (B, T_coarse, 2)
        traj_coarse = traj_coarse.clamp(-1, 1)

        # upsample coarse feature to mid resolution
        traj_mid = interpolate_time(traj_coarse, self.T_mid) # (B, T_mid, 2)

        # ---- Stage 1: Mid residual at T_mid (conditioned on sampled traj feature from upsampled x_coarse_out) ----
        h_mid = interpolate_time(in_feat, self.T_mid) # (B, T_mid, Ch)
        map_mid = sample_map_along_traj(map_cond, traj_mid) # (B, T_mid, map_dim)

        res_at_mid = self.head_mid(torch.cat([h_mid, map_mid], dim=-1), global_cond) # (B, T_mid, 2)

        traj_fine = interpolate_time(res_at_mid, self.T_fine) + interpolate_time(traj_coarse, self.T_fine) # (B, T_fine, 2)
        traj_fine = traj_fine.clamp(-1, 1)

        # ---- Stage 2: Fine residual at T_fine (conditioned on sampled traj feature from upsampled x_mid_out) ----
        map_fine = sample_map_along_traj(map_cond, traj_fine) # (B, T_fine, map_dim)

        res_at_fine = self.head_fine(torch.cat([in_feat, map_fine], dim=-1), global_cond) # (B, T_fine, 2)

        traj_refined = (traj_fine + res_at_fine).clamp(-1, 1) # (B, T_fine, 2)

        # convert trajectory to noise prediction format for diffusion loss calculation
        eps_hat = (x_t_in - sqrt_ab * traj_refined) / (sqrt_omab + 1e-8) # (B, T_fine, 2)

        aux = {
            "traj_coarse": traj_coarse,
            "traj_mid": (res_at_mid + traj_mid).clamp(-1, 1),
            "traj_fine": traj_refined
        }
        return eps_hat, traj_refined, aux
        
