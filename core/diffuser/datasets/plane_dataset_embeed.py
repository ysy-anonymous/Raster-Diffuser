import torch
import numpy as np
from collections import namedtuple

from core.diffuser.datasets.distance_map_gen import distance_to_obstacle, signed_distance_field

# Data : [n, 30, 2]
def get_data_stats(data: torch.Tensor):
    _, _, d = data.shape  # n: batch size, T: sequence length, d: dimension
    flat = data.reshape(-1, d)  # Flatten to [n*T, d]

    data_min = flat.min(dim=0).values  # shape: (d,)
    data_max = flat.max(dim=0).values  # shape: (d,)

    stats = {
        "min": data_min,
        "max": data_max,
    }
    return stats

def normalize_point(data: torch.Tensor, stats):
    eps = 1e-8
    norm = (data - stats['min']) / (stats['max'] - stats['min'] + eps)  # → [0, 1]
    norm = norm * 2 - 1  # → [-1, 1]
    return norm

def normalize_data(data: torch.Tensor, stats):
    n, T, d = data.shape  
    flat = data.reshape(-1, d)


    eps = 1e-8
    norm = (flat - stats['min']) / (stats['max'] - stats['min'] + eps)  # → [0, 1]
    norm = norm * 2 - 1  # → [-1, 1]

    norm = norm.reshape(n, T, d)
    return norm

def denormalize_data(data: torch.Tensor, stats):
    n, T, d = data.shape
    flat = data.reshape(-1, d)

    eps = 1e-8
    denorm = (flat + 1) / 2 * (stats['max'] - stats['min'] + eps) + stats['min']  # → [min, max]

    denorm = denorm.reshape(n, T, d)
    return denorm

class PlanePlanningDataSets(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, include_distance_map=True, sdf_only=False, dmap_only=False, cell_size=1.0, boundary_zero=True, input_upsample=1.0, normalize_st=True):
        self.data = np.load(dataset_path, allow_pickle=True).item()

        self.paths = torch.tensor(self.data['paths'], dtype=torch.float32)           # (N, T, action_dim) -> timesteps, (x, y)
        self.start = torch.tensor(self.data['start'], dtype=torch.float32)           # (N, obs_dim) -> x, y
        self.goal = torch.tensor(self.data['goal'], dtype=torch.float32)             # (N, obs_dim) -> x, y
        self.map = torch.tensor(self.data['map'], dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W) -> binary map
        
        self.include_distance_map = include_distance_map
        self.cell_size=cell_size
        self.boundary_zero=boundary_zero
        self.input_upsample = float(input_upsample)
        self.sdf_only = sdf_only
        self.dmap_only = dmap_only

        if not include_distance_map and sdf_only:
            raise ValueError("set the include_distance_map to True if you want to use sdf_only option, since sdf is computed based on distance map.")
        
        if not include_distance_map and dmap_only:
            raise ValueError("set the include_distance_map to True if you want to use dmap_only option.")
        
        stats = get_data_stats(self.paths)
        self.norm_stats = stats
        # print("Data stats: ", stats)

        self.paths = normalize_data(self.paths, stats) # this will be used for label
        if normalize_st:
            self.start = normalize_point(self.start, stats) # Change: Now start & end point also normalized
            self.goal = normalize_point(self.goal, stats) # Change: Now start & end point also normalized

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        sample = self.paths[idx]  # (T, action_dim)
        map_cond = self.map[idx]  # (1, H, W)
        env_cond = torch.cat([self.start[idx], self.goal[idx]], dim=-1) # (2 * obs_dim)
        
        if self.include_distance_map:
            
            mask = map_cond[None, ...]
            if self.sdf_only:
                sdf_only = signed_distance_field(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
                map_cond = sdf_only # For sdf-only setting, we don't normalize the sdf values.
            elif self.dmap_only:
                dmap_only = distance_to_obstacle(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
                map_cond = dmap_only # For dmap-only setting, we don't normalize the distance map values.
            else:
                dst = distance_to_obstacle(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
                sdf = signed_distance_field(mask, cell_size=self.cell_size, boundary_zero=self.boundary_zero)
                map_cond = torch.cat([mask, dst, sdf], dim=1) # ex) (1, 3, 8, 8)
                map_cond[:, 1:] = map_cond[:, 1:] / (map_cond[:, 1:].amax(dim=(-2, -1), keepdim=True) + 1e-6)
                map_cond = map_cond.squeeze(0) # remove batch dimension ...

            
        # upsample after creating distance field. creating distance field on high resolution map takes huge computation.
        if self.input_upsample != 1.0:
            if len(map_cond.shape) == 3:
                map_cond = map_cond[None, ...] # if no batch dimension, add batch dim
            map_cond = torch.nn.functional.interpolate(map_cond, scale_factor=self.input_upsample, mode='bilinear')
            map_cond = map_cond.squeeze(0) # (3, H, W) or (1, H, W)
        else:    
            if len(map_cond.shape) == 4:
                map_cond = map_cond.squeeze(0) # (3, H, W) or (1, H, W)
        
        
        return {
            "sample": sample,     # label data
            "map": map_cond,      # map data (binary map)
            "env": env_cond,      # start, goal location concatenated
        }


def main():
    dataset = np.load('dataset/train_data_set.npy', allow_pickle=True).item()
    stats = get_data_stats(torch.tensor(dataset['paths'], dtype=torch.float32))
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()
