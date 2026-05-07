import numpy as np
import torch
import argparse

from core.comp_diffusion.cd_stgl_sml_dfu.stgl_sml_policy_v1 import Stgl_Sml_Policy_V1
from core.comp_diffusion.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from core.comp_diffusion.datasets.rrt_map.rrt_test_dataset import RRTTestDataset
from utils.value_utils import show_multiple_with_collision_colors
from utils.load_utils import build_comp_diff_from_cfg

def unnormalize(data, stats):
    eps = 1e-8
    for k, v in stats.items():
        stats[k] = np.array(v)
    denorm = (data + 1) / 2 * (stats['max'] - stats['min'] + eps) + stats['min']  # → [min, max]
    return denorm


def parse_argument():
    parser = argparse.ArgumentParser(description='test diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: Directly Adapted Comp Diffuser, 1: Comp Diffuser V2')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model was trained with DDP (affects how weights are loaded)')
    parser.add_argument('--cp_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion/run2/state_90000.pt', help='trained model weight path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run model')
    parser.add_argument('--num_vis', type=int, default=500, help='number of test scenarios to visualize (if test_num > num_vis, then randomly select num_vis samples for visualization)')
    parser.add_argument('--vis_fname', type=str, default='test_results', help='name of visualization file')
    parser.add_argument('--test_path', type=str, default='datasets/test_scenarios_25000_8x8.npy', help='path to test scenarios npy file')
    parser.add_argument('--test_bs', type=int, default=2048, help='batch size for testing')

    args = parser.parse_args()
    
    return args


# norm_stats: Normalization stats
def generate_path(policy: Stgl_Sml_Policy_V1, start, goal, map_cond, device):
    start_np = start.cpu().numpy()
    goal_np = goal.cpu().numpy()

        
    g_cond = {
        'st_gl': np.array([ # (2, B, dim)
            start_np, # shape (B, 2)
            goal_np # shape (B, 2)
        ], dtype=np.float32)
    }

    m_out_list, pick_traj_acc = policy.gen_cond_stgl_parallel(g_cond=g_cond, map_cond=map_cond, b_s=5)

    return torch.tensor(pick_traj_acc, device=device)


def test_diffusion_policy(policy, test_dataloader, num_vis=100, device='cuda', vis_fname='test_results'):
    """
    Test diffusion policy using value_utils functions
    """

    # Storage for results
    grid_list = []
    path_list = []
    start_list = []
    goal_list = []
    
    print(f"Testing {len(test_dataloader)} scenarios...")
    
    for i, data in enumerate(test_dataloader):
        map_cond = data['map'].to(device) # (1, C, H, W)
        start = data['start'].to(device)
        goal = data['goal'].to(device)
        # Generate path using diffusion policy
        trajectory = generate_path(policy, start, goal, map_cond, device=device) # use this when you normalize start/goal point

        # Convert to numpy for visualization
        grid_np = map_cond[:, 0].cpu().numpy() # [Bs, H, W]
        path_np = trajectory.cpu().numpy() # [Bs, 32, 2]
        start_np = start.cpu().numpy() # [Bs, 2]
        goal_np = goal.cpu().numpy() # [Bs, 2]
        
        # Store results
        grid_list.extend(grid_np) # grid_list.append(grid_np)
        path_list.extend(path_np) # path_list.append(path_np)
        start_list.extend(start_np) # start_list.append(start_np)
        goal_list.extend(goal_np) # goal_list.append(goal_np)
        
        print(f"Test {i+1}/{len(test_dataloader)} completed")
    
    print(f"Generated {len(path_list)} valid test cases")
    # Visualize results with collision-based coloring
    indices = list(range(len(path_list)))
    results = show_multiple_with_collision_colors(
        grid_list, path_list, start_list, goal_list, indices, cols=5, num_vis=num_vis, vis_fname=vis_fname
    )
    
    return results


def main():
    """
    Main function to load model and run tests
    """
    
    # parse argument
    args = parse_argument()
    
    # Model configuration
    test_model = args.model_id
    is_ddp_trained = args.ddp_trained
    ckpt_path = args.cp_path
    device = args.device
    num_vis = args.num_vis
    vis_fname = args.vis_fname
    test_path = args.test_path
    test_bs = args.test_bs

    if is_ddp_trained:
        print("Testing DDP-trained model...")
        ddp=True
    else:
        print("Testing non-DDP-trained model...")
        ddp=False
    

    print("Loading diffusion policy model...")
    diffusion_model, config_dict = build_comp_diff_from_cfg(test_model, ddp=ddp)
    diffusion_model = diffusion_model.to(device)
    print("Model loaded successfully!")


    # ===== Build Dataset Normalizer ===== #
    norm_stats = config_dict['normalizer'] # dataset norm stats
    data_dict = {}
    # Fake Actions
    data_dict['actions'] = np.array([norm_stats['min'], norm_stats['max']], dtype=np.float32) # our task does not use 'actions'
    data_dict['observations'] = np.array([norm_stats['min'], norm_stats['max']], dtype=np.float32)
    normalizer = LimitsNormalizer
    # normalizer = LimitsNormalizer(min=np.array(norm_stats['min'], dtype=np.float32), max=np.array(norm_stats['max'], dtype=np.float32))
    dataset_normalizer = DatasetNormalizer(data_dict, normalizer, eval_solo=True, path_lengths=None)
    
    policy = Stgl_Sml_Policy_V1(
        diffusion_model=diffusion_model,
        normalizer=dataset_normalizer,
        **config_dict['policy_config']
    )
    
    # Load weights
    policy.load_weights(ckpt_path)
    print("Model loaded successfully!")

    test_dataset = RRTTestDataset(dataset_path=test_path, normalize_mode='grid',
                                  coord_min=0.0, coord_max=8.0, include_distance_map=config_dict['dataset']['include_distance_map'],
                                  cell_size=config_dict['dataset']['cell_size'], boundary_zero=config_dict['dataset']['boundary_zero'],
                                  input_upsample=config_dict['dataset']['input_upsample']) # normalize start/goal point for testing
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)
    
    # # here is subset
    # sample_indices = []
    # dataset_len = len(test_dataset)
    # batch_size = test_bs
    # range_of_idx = range(0, 10, 1) # visualize first 500 samples for visualization
    # for batch_idx in range_of_idx:
    #     start_idx = batch_idx * batch_size
    #     end_idx = min((batch_idx + 1) * batch_size, dataset_len)
    #     sample_indices.extend(range(start_idx, end_idx))
    
    # # 3. Create a subset containing only the requested data
    # subset = torch.utils.data.Subset(test_dataset, sample_indices)

    # # 4.Spin up a temporary, highly targeted dataloader
    # fast_dataloader = torch.utils.data.DataLoader(
    #     subset,
    #     batch_size=batch_size,
    #     shuffle=False, # Force sequential reading for this specific range
    #     num_workers=getattr(test_dataloader, "num_workers", 0),
    #     collate_fn=getattr(test_dataloader, "collate_fn", None),
    #     pin_memory=getattr(test_dataloader, "pin_memory", False)
    # )

    # Run tests
    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(policy, test_dataloader=test_dataloader, num_vis=num_vis, device=device, vis_fname=vis_fname)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    results = main()