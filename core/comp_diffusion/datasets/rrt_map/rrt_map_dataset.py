import torch
import numpy as np
from .normalization import get_data_stats
from core.comp_diffusion.datasets.rrt_map.distance_map_gen import (
    signed_distance_field,
    distance_to_obstacle,
)

class RRTCompDataset(torch.utils.data.Dataset):
    """
    Paper-faithful CompDiffuser dataset adaptation for 2D-map planning.

    Key idea:
      - training label is still a short chunk
      - task start/goal are NOT injected as per-window local env conditions
      - global start/goal are returned only for boundary-conditioning logic
      - env is zeroed so the added obs-encoder path is disabled
    """

    def __init__(
        self,
        dataset_path: str,
        horizon=16,
        stride=2,
        include_map=True,
        normalize_mode="grid",
        coord_min=0.0,
        coord_max=8.0,
        include_distance_map=True,
        cell_size=1.0,
        boundary_zero=True,
        input_upsample=1.0,
    ):
        self.data = np.load(dataset_path, allow_pickle=True).item()
        self.paths = torch.tensor(self.data["paths"], dtype=torch.float32)          # (N, T, 2)
        self.start = torch.tensor(self.data["start"], dtype=torch.float32)          # (N, 2)
        self.goal = torch.tensor(self.data["goal"], dtype=torch.float32)            # (N, 2)
        self.map = torch.tensor(self.data["map"], dtype=torch.float32).unsqueeze(1) # (N, 1, 8, 8)

        self.horizon = int(horizon)
        self.stride = int(stride)
        self.include_map = bool(include_map)
        self.normalize_mode = normalize_mode
        self.include_distance_map = bool(include_distance_map)
        self.cell_size = float(cell_size)
        self.boundary_zero = bool(boundary_zero)
        self.input_upsample = float(input_upsample)

        if not self.include_map:
            raise ValueError("include_map must be True for this task.")
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")

        if normalize_mode == "grid":
            self.coord_min = torch.tensor([coord_min, coord_min], dtype=torch.float32)
            self.coord_max = torch.tensor([coord_max, coord_max], dtype=torch.float32)
        elif normalize_mode == "dataset":
            stats = get_data_stats(self.paths)
            self.coord_min = stats["min"]
            self.coord_max = stats["max"]
        elif normalize_mode is None:
            self.coord_min = None
            self.coord_max = None
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

        self.indices = []
        for traj_idx in range(len(self.paths)):
            T = self.paths[traj_idx].shape[0]
            if T < self.horizon:
                continue
            for ws in range(0, T - self.horizon + 1, self.stride):
                we = ws + self.horizon
                self.indices.append((traj_idx, ws, we, T))

        if len(self.indices) == 0:
            raise ValueError("No valid windows found.")

    def __len__(self):
        return len(self.indices)

    def _normalize_points(self, pts: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode is None:
            return pts.to(torch.float32)
        denom = self.coord_max - self.coord_min
        denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
        pts01 = (pts - self.coord_min) / denom
        pts11 = pts01 * 2.0 - 1.0
        return pts11.to(torch.float32)

    def _build_map_cond(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.to(torch.float32)  # (1, 8, 8)

        if self.include_distance_map:
            mask = grid.unsqueeze(0)  # (1, 1, 8, 8)
            dst = distance_to_obstacle(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
            sdf = signed_distance_field(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
            map_cond = torch.cat([mask, dst, sdf], dim=1)  # (1, 3, 8, 8)

            # keep your current per-sample scaling for now, to avoid changing two things at once
            map_cond[:, 1:] = map_cond[:, 1:] / (
                map_cond[:, 1:].amax(dim=(-2, -1), keepdim=True) + 1e-6
            )
            map_cond = map_cond.squeeze(0)  # (3, 8, 8)
        else:
            map_cond = grid  # (1, 8, 8)

        if self.input_upsample != 1.0:
            if map_cond.ndim == 3:
                map_cond = map_cond.unsqueeze(0)
            map_cond = torch.nn.functional.interpolate(
                map_cond, scale_factor=self.input_upsample, mode="bilinear", align_corners=False
            ).squeeze(0)

        return map_cond

    def __getitem__(self, idx):
        traj_idx, ws, we, T = self.indices[idx]

        chunk_raw = self.paths[traj_idx][ws:we]          # (H, 2)
        chunk = self._normalize_points(chunk_raw)

        global_start = self._normalize_points(self.start[traj_idx:traj_idx+1])[0]
        global_goal = self._normalize_points(self.goal[traj_idx:traj_idx+1])[0]

        is_first_chunk = (ws == 0)
        is_last_chunk = (we == T)

        map_cond = self._build_map_cond(self.map[traj_idx])

        return {
            "sample": chunk,                     # (H, 2), normalized
            "map": map_cond,                    # (C, Hm, Wm)
            "task_start": global_start,         # normalized global start
            "task_goal": global_goal,           # normalized global goal
            "boundary_mask": torch.tensor(
                [float(is_first_chunk), float(is_last_chunk)],
                dtype=torch.float32
            ),
            "meta": {
                "traj_idx": traj_idx,
                "window_start": ws,
                "window_end": we,
                "traj_len": T,
            },
        }


def rrt_comp_collate_fn(batch):
    return {
        "sample": torch.stack([x["sample"] for x in batch], dim=0),
        "map": torch.stack([x["map"] for x in batch], dim=0),
        "task_start": torch.stack([x["task_start"] for x in batch], dim=0),
        "task_goal": torch.stack([x["task_goal"] for x in batch], dim=0),
        "boundary_mask": torch.stack([x["boundary_mask"] for x in batch], dim=0),
        "meta": [x["meta"] for x in batch],
    }