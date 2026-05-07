# Advanced RND Network that considers the obstacle, stgl condition information.
# In order to reinforce the conditioning ability of trajectory shape encoder, we adapt FiLM layer here also.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class FiLM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x, cond):
        # x: (B, C, L) or (B, C, H, W)
        # cond: (B, D_cond)
        gamma_beta = self.linear(cond)  # (B, 2 * out_dim)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # each (B, out_dim)
        gamma = gamma.unsqueeze(-1)  # (B, out_dim, 1) for Conv1d or (B, out_dim, 1, 1) for Conv2d
        beta = beta.unsqueeze(-1)
        return x * gamma + beta
    
class Conv1DFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.film = FiLM(in_dim=cond_dim, out_dim=out_channels)

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.film(x, cond)
        return F.leaky_relu(x)

class FiLMConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = Conv1DFiLM(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, cond):
        x = self.conv(x, cond)
        x = self.act(x)
        return x


class TrajEncoder(nn.Module):
    def __init__(self, in_channels, cond_dim, hidden_dim=256, out_dim=256):
        super().__init__()
        self.block1 = FiLMConvBlock(in_channels, 64, cond_dim, 4, 2, 1)
        self.block2 = FiLMConvBlock(64, 128, cond_dim, 4, 2, 1)
        self.block3 = FiLMConvBlock(128, hidden_dim, cond_dim, 4, 2, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x, cond):
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out_act(x)
        return x


class MaskEncoder(nn.Module):
    # If you use the distance_map also, then in_channels would be 3 instead of 1
    def __init__(self, in_channels=1, base_dim=32, out_dim=256):
        super().__init__()

        self.net = nn.Sequential( # Input map is only 8x8, so we don't downsample here.
            nn.Conv2d(in_channels, base_dim, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(base_dim * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        return self.net(x)


class ContextEncoder(nn.Module):
    # Start/goal condition encoder, in_dim=4 for (xs, ys, xg, yg)
    def __init__(self, in_dim=4, hidden_dim=128, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, D_ctx), e.g. start/goal = (xs, ys, xg, yg)
        return self.net(x)


class FusionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalRNDModel(nn.Module):
    def __init__(
        self,
        traj_channels,
        rnd_output_dim=256,
        context_dim=4,
        mask_in_channels=1,
        traj_feat_dim=128,
        mask_feat_dim=128,
        ctx_feat_dim=64,
    ):
        super().__init__()

        # Predictor encoders
        self.pred_traj_encoder = TrajEncoder(
            in_channels=traj_channels,
            cond_dim=mask_feat_dim + ctx_feat_dim, # FiLM conditioning on both mask and context
            hidden_dim=traj_feat_dim,
            out_dim=traj_feat_dim,
        )
        self.pred_mask_encoder = MaskEncoder(
            in_channels=mask_in_channels,
            base_dim=32,
            out_dim=mask_feat_dim,
        )
        self.pred_ctx_encoder = ContextEncoder(
            in_dim=context_dim,
            hidden_dim=ctx_feat_dim,
            out_dim=ctx_feat_dim,
        )
        self.predictor_head = FusionMLP(
            in_dim=traj_feat_dim + mask_feat_dim + ctx_feat_dim,
            out_dim=rnd_output_dim,
        )

        # Target encoders
        self.tgt_traj_encoder = TrajEncoder(
            in_channels=traj_channels,
            cond_dim=mask_feat_dim + ctx_feat_dim, # FiLM conditioning on both mask and context
            hidden_dim=traj_feat_dim,
            out_dim=traj_feat_dim,
        )
        self.tgt_mask_encoder = MaskEncoder(
            in_channels=mask_in_channels,
            base_dim=32,
            out_dim=mask_feat_dim,
        )
        self.tgt_ctx_encoder = ContextEncoder(
            in_dim=context_dim,
            hidden_dim=ctx_feat_dim,
            out_dim=ctx_feat_dim,
        )
        self.target_head = FusionMLP(
            in_dim=traj_feat_dim + mask_feat_dim + ctx_feat_dim,
            out_dim=rnd_output_dim,
        )

        self._init_weights()
        self._freeze_target()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_target(self):
        for param in (
            list(self.tgt_traj_encoder.parameters()) +
            list(self.tgt_mask_encoder.parameters()) +
            list(self.tgt_ctx_encoder.parameters()) +
            list(self.target_head.parameters())
        ):
            param.requires_grad = False

    def forward(self, traj, mask, start_goal):
        """
        traj:       (B, C_traj, L)
        mask:       (B, 1, H, W)
        start_goal: (B, 4) or (B, context_dim)
        """
        # Predictor
        pred_mask = self.pred_mask_encoder(mask)
        pred_ctx = self.pred_ctx_encoder(start_goal)
        pred_cond = torch.cat([pred_mask, pred_ctx], dim=-1)
        pred_traj = self.pred_traj_encoder(traj, cond=pred_cond)
        pred_feat = torch.cat([pred_traj, pred_mask, pred_ctx], dim=-1)
        pred_feat = self.predictor_head(pred_feat)

        # Target
        with torch.no_grad():
            tgt_mask = self.tgt_mask_encoder(mask)
            tgt_ctx = self.tgt_ctx_encoder(start_goal)
            tgt_cond = torch.cat([tgt_mask, tgt_ctx], dim=-1)
            tgt_traj = self.tgt_traj_encoder(traj, cond=tgt_cond)
            tgt_feat = torch.cat([tgt_traj, tgt_mask, tgt_ctx], dim=-1)
            tgt_feat = self.target_head(tgt_feat)

        # Per-sample RND error
        rnd_error = torch.norm(pred_feat - tgt_feat, p=2, dim=-1)
        return rnd_error