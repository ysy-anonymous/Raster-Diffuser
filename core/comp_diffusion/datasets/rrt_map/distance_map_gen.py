import torch

# continuous map that has distance-to-obstacle map (Euclidean distance)
def _pairwise_grid_dist(H: int, W:int, device, dtype):
    # coords: (HW, 2) with (y, x)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    coords = torch.stack([ys, xs], dim=-1).view(-1, 2) # (HW, 2)
    # D[p, q] = ||coords[p] - coords[q]||_2
    diff = coords[:, None, :] - coords[None, :, :] # (HW, HW, 2)
    D = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12) # (HW, HW)
    return D

def distance_to_obstacle(mask: torch.Tensor, cell_size: float = 1.0, boundary_zero: bool = False):
    """
        mask: (B, 1, H, W) with 1=True obstacle, 0=False free
        returns: dist_out (B, 1, H, W) distance from each cell center to nearest obstacle cell center
         multiplied by cell_size (meters if cell_size is meters/cell)
        boundary_zero: if True, subtract 0.5* cell_size so adjacent-to-obstacle becomes ~0 at boundary
    """
    # print('mask.shape: ', mask.shape)
    assert mask.ndim == 4 and mask.size(1) == 1
    
    B, _, H, W = mask.shape
    device = mask.device
    # use float dtype for distance
    dtype = torch.float32 if mask.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else mask.dtype
    mask_f = mask.to(dtype=torch.float32)
    
    # calculate pairwise grid distance
    HW = H * W
    D = _pairwise_grid_dist(H, W, device=device, dtype=torch.float32) # (HW, HW)
    
    obs = (mask_f.view(B, HW) > 0.5).to(torch.float32)
    has_obs = obs.sum(dim=-1) > 0
    
    # penalty is huge for non-obstacle columns q so min ignores them
    max_dist = (torch.sqrt(torch.tensor((H-1) ** 2 + (W-1) ** 2, device=device, dtype=torch.float32)) * cell_size).item()
    BIG = max_dist * 1e3
    
    penalty = (1.0 - obs) * BIG   # (B, HW)
    D_b = D.unsqueeze(0) + penalty[:, None, :]  # (B, HW, HW)
    dist = D_b.min(dim=-1).values # (B, HW)
    
    # handle batches with no obstacles
    dist[~has_obs] = max_dist
    
    dist = dist.view(B, 1, H, W) * cell_size
    
    if boundary_zero:
        # If you interpret each occuped cell as a square, boundary is ~0.5 cell away from center.
        dist = torch.clamp(dist - 0.5 * cell_size, min=0.0)
    
    return dist.to(dtype)


# Signed Distance Field (SDF): positive in free space, negative inside obstacles
def signed_distance_field(mask: torch.Tensor, cell_size: float=1.0, boundary_zero: bool =False):
    """
    SDF > 0 in free space, SDF < 0 in obstacle cells.
    Magnitude is distance to nearest boundary (approx, center-to-center)
    """
    assert mask.ndim == 4 and mask.size(1) == 1
    B, _, H, W = mask.shape
    device = mask.device
    mask_f = (mask > 0.5) # bool
    
    # distance from every cell to nearest obstacle (for free space)
    dist_out = distance_to_obstacle(mask_f.float(), cell_size=cell_size, boundary_zero=boundary_zero)
    
    # distance from every cell to nearest free cell (for inside obstacle)
    dist_in = distance_to_obstacle((~mask_f).float(), cell_size=cell_size, boundary_zero=boundary_zero)
    
    sdf = dist_out.clone()
    sdf[mask_f] = -dist_in[mask_f] # negative inside obstacles
    return sdf


def main():
    mask = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]]])
    print("mask.shape: ", mask.shape)
    dis = distance_to_obstacle(mask, cell_size=0.5)

    print("mask: ", mask)
    print("distance: ", dis)

    sdf = signed_distance_field(mask, cell_size=0.5, boundary_zero=True)
    print("sdf: ", sdf)

if __name__ == '__main__':
    main()
