import torch
import numpy as np
from collections import namedtuple

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


def normalize_point_np(data, stats):
    eps = 1e-8
    data = np.array(data)
    for k, v in stats.items():
        stats[k] = np.array(v)
    norm = (data - stats['min']) / (stats['max'] - stats['min'] + eps)  # [0, 1]
    norm = norm * 2 - 1  # [-1, 1]
    return norm
