import torch
import numpy as np
from .normalization import get_data_stats
from core.comp_diffusion.datasets.rrt_map.distance_map_gen import (
    signed_distance_field,
    distance_to_obstacle,
)

class RRTTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        normalize_mode="grid",
        coord_min=0.0,
        coord_max=8.0,
        include_distance_map=True,
        cell_size=1.0,
        boundary_zero=True,
        input_upsample=1.0,
    ):
        self.data = np.load(dataset_path, allow_pickle=True).item()

        self.paths = torch.tensor(self.data["paths"], dtype=torch.float32)
        self.start = torch.tensor(self.data["start"], dtype=torch.float32)          # raw
        self.goal = torch.tensor(self.data["goal"], dtype=torch.float32)            # raw
        self.map = torch.tensor(self.data["map"], dtype=torch.float32).unsqueeze(1)

        self.include_distance_map = bool(include_distance_map)
        self.cell_size = float(cell_size)
        self.boundary_zero = bool(boundary_zero)
        self.input_upsample = float(input_upsample)
        self.normalize_mode = normalize_mode

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

    def __len__(self):
        return len(self.paths)

    def _normalize_points(self, pts: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode is None:
            return pts.to(torch.float32)
        denom = self.coord_max - self.coord_min
        denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
        pts01 = (pts - self.coord_min) / denom
        pts11 = pts01 * 2.0 - 1.0
        return pts11.to(torch.float32)

    def _build_map_cond(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.to(torch.float32)

        if self.include_distance_map:
            mask = grid.unsqueeze(0)
            dst = distance_to_obstacle(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
            sdf = signed_distance_field(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
            map_cond = torch.cat([mask, dst, sdf], dim=1)
            map_cond[:, 1:] = map_cond[:, 1:] / (
                map_cond[:, 1:].amax(dim=(-2, -1), keepdim=True) + 1e-6
            )
            map_cond = map_cond.squeeze(0)
        else:
            map_cond = grid

        if self.input_upsample != 1.0:
            if map_cond.ndim == 3:
                map_cond = map_cond.unsqueeze(0)
            map_cond = torch.nn.functional.interpolate(
                map_cond, scale_factor=self.input_upsample, mode="bilinear", align_corners=False
            ).squeeze(0)

        return map_cond

    def __getitem__(self, idx):
        map_cond = self._build_map_cond(self.map[idx])
        sample = self._normalize_points(self.paths[idx])

        return {
            "sample": sample,          # normalized full trajectory, eval/vis only
            "map": map_cond,
            "start": self.start[idx],  # raw start
            "goal": self.goal[idx],    # raw goal
        }