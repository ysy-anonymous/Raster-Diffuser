from collections import namedtuple
import torch
import numpy as np
import einops
import pdb

import core.rediffuser.networks.diffuser.utils as utils
from core.rediffuser.datasets.distance_map_gen import distance_to_obstacle, signed_distance_field

Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:

    def __init__(self, diffusion_model, config, horizon, cool_term=1.0):
        self.diffusion_model = diffusion_model
        self.config = config # configuration file
        self.horizon = horizon
        self.diffusion_model.cool_term = cool_term

    @property
    def device(self):
        return next(self.diffusion_model.parameters()).device

    def _to_tensor(self, x, device=None, dtype=torch.float32):
        """
        Convert input to torch.Tensor on the target device.
        Handles torch.Tensor, numpy arrays, lists, scalars.
        """
        if device is None:
            device = self.device

        if x is None:
            return None

        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)

        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device, dtype=dtype)

        return torch.tensor(x, device=device, dtype=dtype)

    def _to_tensor_dict(self, data_dict, device=None, dtype=torch.float32):
        """
        Convert all values in a stats dict to tensors on the same device.
        """
        if device is None:
            device = self.device

        out = {}
        for k, v in data_dict.items():
            out[k] = self._to_tensor(v, device=device, dtype=dtype)
        return out

    def normalize_point(self, data, stats, device=None):
        eps = 1e-8
        if device is None:
            device = self.device

        data = self._to_tensor(data, device=device, dtype=torch.float32)
        stats = self._to_tensor_dict(stats, device=device, dtype=torch.float32)

        norm = (data - stats['min']) / (stats['max'] - stats['min'] + eps)  # [0, 1]
        norm = norm * 2 - 1  # [-1, 1]
        return norm

    def denormalize_data(self, data, stats, device=None):
        if device is None:
            device = data.device

        data = self._to_tensor(data, device=device, dtype=torch.float32)
        stats = self._to_tensor_dict(stats, device=device, dtype=torch.float32)

        n, T, d = data.shape
        flat = data.reshape(-1, d)

        eps = 1e-8
        denorm = (flat + 1) / 2 * (stats['max'] - stats['min'] + eps) + stats['min']
        denorm = denorm.reshape(n, T, d)
        return denorm

    def _format_conditions(self, map_cond, st_gl, stats, device, batch_size):
        """
        Expected raw inputs:
            st_gl: [4] or [1, 4]
            map_cond: [C, H, W] or [1, C, H, W]
        Returns:
            map_cond: [B, C, H, W]
            obs_cond: [B, 4]
        """
        start = st_gl[..., :2]
        goal = st_gl[..., 2:]
        start_norm = self.normalize_point(start, stats, device=device)
        goal_norm = self.normalize_point(goal, stats, device=device)
        obs_normed = torch.cat((start_norm, goal_norm), dim=-1)

        if obs_normed.ndim == 1:
            obs_normed = obs_normed.unsqueeze(0)   # [1, D]
        elif obs_normed.ndim == 2:
            pass
        else:
            raise Exception(f"Invalid st_gl shape: {obs_normed.shape}")

        map_cond = self._to_tensor(map_cond, device=device, dtype=torch.float32)

        if map_cond.ndim == 3:
            map_cond = map_cond.unsqueeze(0)       # [1, C, H, W]
        elif map_cond.ndim == 4:
            pass
        else:
            raise Exception(f"Invalid map_cond shape: {map_cond.shape}")
        
        if self.config['dataset']['include_distance_map']:
            cell_size = self.config['dataset']['cell_size']
            boundary_zero = self.config['dataset']['boundary_zero']
            dst = distance_to_obstacle(map_cond, cell_size=cell_size, boundary_zero=boundary_zero)
            sdf = signed_distance_field(map_cond, cell_size=cell_size, boundary_zero=boundary_zero)
            map_cond = torch.cat([map_cond, dst, sdf], dim=1)

            # Normalize continuous map, signed distance map
            map_cond[:, 1:] = map_cond[:, 1:] / (map_cond[:, 1:].amax(dim=(-2, -1), keepdim=True) + 1e-6)
        else:
            map_cond = map_cond

        obs_cond = obs_normed.repeat_interleave(batch_size, dim=0)
        map_cond = map_cond.repeat_interleave(batch_size, dim=0)

        return map_cond, obs_cond

    def __call__(self, map_cond, st_gl, device=None, stats=None, batch_repeat=1, dataset_mode=False):
        """
        when dataset_mode=True:
            expects already batched dataset-style inputs
                map_cond: [B, C, H, W]
                st_gl:    [B, D]
        when dataset_mode=False:
            expects raw numpy/tensor inputs
                map_cond: [C, H, W] or [1, C, H, W]
                st_gl:    [D] or [1, D]
        """
        if device is None:
            device = self.device
        else:
            device = torch.device(device)
        if len(map_cond.shape) == 3:
            batch_before_expansion = 1
        elif (len(map_cond.shape) == 4):
            batch_before_expansion = map_cond.shape[0]

        self.diffusion_model = self.diffusion_model.to(device)
        self.diffusion_model.eval()

        if dataset_mode is False:
            if stats is None:
                raise ValueError("stats must be provided when dataset_mode=False")
            map_cond, obs_cond = self._format_conditions(
                map_cond, st_gl, stats, device, batch_repeat
            )
        else:
            map_cond = self._to_tensor(map_cond, device=device, dtype=torch.float32)
            obs_cond = self._to_tensor(st_gl, device=device, dtype=torch.float32)

            # Repeat batch dimension to generate alternative diffusion trajectory
            map_cond = map_cond.repeat_interleave(batch_repeat, dim=0)
            obs_cond = obs_cond.repeat_interleave(batch_repeat, dim=0)


            if map_cond.ndim != 4:
                raise Exception(f"Invalid dataset_mode map_cond shape: {map_cond.shape}")
            if obs_cond.ndim != 2:
                raise Exception(f"Invalid dataset_mode st_gl shape: {obs_cond.shape}")

        with torch.no_grad():
            sample = self.diffusion_model(map_cond, obs_cond, horizon=self.horizon)

        if stats is not None:
            trajectory = self.denormalize_data(sample, stats, device=device)
        else:
            trajectory = sample

        return trajectory

    def load_weights(self, ckpt_path: str, device=None, use_ema=True):
        if device is None:
            device = self.device if any(True for _ in self.diffusion_model.parameters()) else torch.device("cpu")
        else:
            device = torch.device(device)

        state_dict = torch.load(ckpt_path, map_location=device)

        if use_ema and 'ema' in state_dict:
            self.diffusion_model.load_state_dict(state_dict['ema'])
        elif 'model' in state_dict:
            self.diffusion_model.load_state_dict(state_dict['model'])
        else:
            self.diffusion_model.load_state_dict(state_dict)

        self.diffusion_model = self.diffusion_model.to(device)
        self.diffusion_model.eval()

        step = state_dict.get('step', None)
        return step