import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenMapAttn(nn.Module):
    def __init__(self, token_dim, patch_dim, nhead=4):
        super().__init__()
        self.q = nn.Linear(token_dim, token_dim)
        self.k = nn.Linear(patch_dim, token_dim)
        self.v = nn.Linear(patch_dim, token_dim)
        self.attn = nn.MultiheadAttention(token_dim, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 4*token_dim),
                                nn.SiLU(), nn.Linear(4*token_dim, token_dim))
    
    def forward(self, z, patches):
        # z: (B, K, D)
        # patches: (B, N, Cm)
        q = self.q(z)
        k = self.k(patches)
        v = self.v(patches)
        out, _ = self.attn(q, k, v)
        z = z + out
        z = z + self.ff(z)
        return z

def sample_map_along_traj(M, x):
    # M: (B, C, H, W), x: (B, K, T, 2) in [-1, 1]
    B, K, T, _ = x.shape
    xg = x.view(B*K, T, 1, 2)
    Mg = M[:, None].expand(B, K, *M.shape[1:]).reshape(B*K, *M.shape[1:])
    feat = F.grid_sample(Mg, xg, mode='bilinear', align_corners=False)
    feat = feat.squeeze(-1).transpose(1, 2)
    return feat.view(B, K, T, -1)

class MultiHypothesisRefiner(nn.Module):
    def __init__(self, K, token_dim, cm, ch, dt, do, R=3):
        super().__init__()
        self.K, self.R = K, R
        self.slot = nn.Parameter(torch.randn(K, token_dim) * 0.02)
        
        self.cond_proj = nn.Linear(cm + dt + do, token_dim)
        self.token_map = TokenMapAttn(token_dim, cm)
        
        # token -> FiLM for h_base
        self.to_film = nn.Linear(token_dim, 2 * ch)
        
        # pool sampled traj-map feats -> token update
        self.traj_pool = nn.Sequential(nn.Linear(cm, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.token_upd = nn.GRUCell(token_dim, token_dim)
        
        # decode x0 (or eps) per hypothesis
        self.head = nn.Sequential(nn.Linear(ch, ch), nn.SiLU(), nn.Linear(ch, 2))
        
        # # predict best hypothesis that will be used for training and inference
        # global_dim = cm + dt + do
        # self.hypo_selector = nn.Sequential(
        #     nn.Linear(token_dim + global_dim + cm, 128),
        #     nn.SiLU(),
        #     nn.Linear(128, 1)
        # )
        
    def forward(self, h_base, M, global_cond, x_t_in, alphas_cumprod, timesteps):
        # h_base: (B, T, Ch) shared trunk features from x_t
        B, T, Ch = h_base.shape
        H, W = M.shape[-2:]
        
        # init tokens
        z = self.slot[None].expand(B, -1, -1) + self.cond_proj(global_cond)[:, None, :]
        
        # flatten map patches
        patches = M.flatten(2).transpose(1, 2) # (B, N, Cm)
        
        # gather alpha_bar(t) for x0 conversion
        alpha_bar = alphas_cumprod[timesteps].view(B, 1, 1)
        sqrt_ab = alpha_bar.sqrt()
        sqrt_omab = (1 - alpha_bar).sqrt()
        
        # iterative refinement
        x0 = None
        for _ in range(self.R):
            z = self.token_map(z, patches) # (B, K, D)
            
            # token-specific FiLM on shared trunk
            film = self.to_film(z) # (B, K, 2Ch)
            gamma, beta = film[..., :Ch], film[..., Ch:] # (B, K, Ch)
            h_k = gamma[:, :, None, :] * h_base[:, None, :, :] + beta[:, :, None, :] # (B, K, T, Ch)
                        
            # decoder eps_k then x0_k (diffusion-consistent coordinate refinement)
            eps_k = self.head(h_k) # (B, K, T, 2)
            x0 = (x_t_in[:, None, :, :] - sqrt_omab[:, None, :, :] * eps_k) / (sqrt_ab[:, None, :, :] + 1e-8)
            x0 = x0.clamp(-1, 1)
            
            # sample map along provisional x0 and update tokens
            samp = sample_map_along_traj(M, x0) # (B, K, T, Cm)
            s = samp.mean(dim=2) # (B, K, Cm)
            upd_in = self.traj_pool(s) # (B, K, D)
            
            # GRUCell expects (B*K, D)
            z = self.token_upd(upd_in.reshape(B*self.K, -1), z.reshape(B*self.K, -1)).view(B, self.K, -1)
        
        # # predict best hypothesis based on neural network prediction
        # g = global_cond[:, None, :].expand(B, self.K, global_cond.size(-1))
        # sel_in = torch.cat([z, s, g], dim=-1)             # (B,K,token_dim+cm+global_dim)
        # k_logits = self.hypo_selector(sel_in).squeeze(-1)   # (B,K)
        
        # return K eps predictions (or mix later)
        eps_k = (x_t_in[:, None, :, :] - sqrt_ab[:, None, :, :] * x0) / (sqrt_omab[:, None, :, :] + 1e-8)
        
        return eps_k, x0
        # return eps_k, x0, k_logits