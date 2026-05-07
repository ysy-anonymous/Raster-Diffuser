import numpy as np
from utils.value_utils import create_test_scenario, show_multiple_with_collision_colors
from utils.load_utils import build_comp_diff_from_cfg

# Diffusion Policy for testing samples...
from core.comp_diffusion.cd_stgl_sml_dfu.stgl_sml_policy_v1 import Stgl_Sml_Policy_V1
from core.comp_diffusion.datasets.normalization import DatasetNormalizer, LimitsNormalizer
from core.comp_diffusion.datasets.rrt_map.distance_map_gen import distance_to_obstacle, signed_distance_field

import torch
import argparse
import random


# norm_stats: Normalization stats
def generate_path(policy: Stgl_Sml_Policy_V1, start, goal, obstacles, config: dict, device):
    
    # ====== Generate Trajectory ====== #
    start_np = start.cpu().numpy()[0]; goal_np = goal.cpu().numpy()[0]
        
    g_cond = {
        'st_gl': np.array([ # (2, n_probs, dim), here n_probs=1.
            [start_np], # shape (1, 2)
            [goal_np] # shape (1, 2)
        ], dtype=np.float32)
    }
    
    # Fake observation sample (all zeros)
    obs_mask = obstacles[0].cpu().numpy()[None, ...]  # [1, 1, 8] -> shape: [1, 1, 8, 8], add batch dim
    obs_mask = torch.tensor(obs_mask, dtype=torch.float32)
    
    if config['dataset']['include_distance_map']:
        cell_size = config['dataset']['cell_size']
        boundary_zero = config['dataset']['boundary_zero']
        dst = distance_to_obstacle(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
        sdf = signed_distance_field(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
        map_cond = torch.cat([obs_mask, dst, sdf], dim=1) # (B, 3, 8, 8)
        
        # Normalize continuous map, signed distance map
        map_cond[:, 1:] = map_cond[:, 1:] / (map_cond[:, 1:].amax(dim=(-2, -1), keepdim=True) + 1e-6)
    else:
        map_cond = obs_mask

    # Currently, b_s=1. This is the repetition count per problem before calling the diffusion sampler.
    # with b_s=1, each problem only gets one sample path through the sampler input batch.
    m_out_list, pick_traj_acc = policy.gen_cond_stgl_parallel(g_cond=g_cond, map_cond=map_cond, b_s=1)
    pick_traj = pick_traj_acc[0]
    
    print("shape of pick_traj: ", pick_traj.shape)
    
    return torch.tensor(pick_traj, device=device).unsqueeze(0) # [1, 36, 2]


def parse_argument():
    parser = argparse.ArgumentParser(description='test diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: Directly Adapted Comp Diffuser, 1: Comp Diffuser V2')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model to test is trained with ddp or not')
    parser.add_argument('--cp_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion/run2/state_90000.pt', help='model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run model')
    parser.add_argument('--test_num', type=int, default=100, help='number of test samples to generate')
    parser.add_argument('--map_size', type=int, default=8, help='test set map size')
    parser.add_argument('--num_vis', type=int, default=100, help='test samples for visualization')
    parser.add_argument('--vis_fname', type=str, default='test_results', help='name of visualization file')
    args = parser.parse_args()
    
    return args


# Test Trained Models
def test_diffusion_policy(policy, config_dict, num_tests=20, map_size=8, num_vis=100, device='cuda', vis_fname='test_results'):
    """
    Test diffusion policy using value_utils functions
    """

    # Test parameters
    bounds = np.array([[0.0, map_size], [0.0, map_size]])
    cell_size = 1.0
    origin = bounds[:, 0]
    max_rectangles = (3 * (map_size//8), 5 * (map_size//8))
    rng = np.random.default_rng(40)
    
    # Storage for results
    grid_list = []
    path_list = []
    start_list = []
    goal_list = []
    
    print(f"Generating {num_tests} test scenarios...")
    
    for i in range(num_tests):
    # Create test scenario using your value_utils function
        start, goal, obstacles = create_test_scenario(
            bounds=bounds,
            cell_size=cell_size,
            origin=origin,
            rng=rng,
            max_rectangles=max_rectangles,
            device=device
        )
        
        # Generate path using diffusion policy
        trajectory = generate_path(policy, start, goal, obstacles, config_dict, device)
                
        # Convert to numpy for visualization
        grid_np = obstacles[0, 0].cpu().numpy()
        path_np = trajectory[0].cpu().numpy()
        start_np = start[0].cpu().numpy()
        goal_np = goal[0].cpu().numpy()
        
        # Store results
        grid_list.append(grid_np)
        path_list.append(path_np)
        start_list.append(start_np)
        goal_list.append(goal_np)
        
        print(f"Test {i+1}/{num_tests} completed")
    
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
    ckpt_path = args.cp_path
    device = args.device
    test_model = args.model_id
    test_num = args.test_num
    map_size = args.map_size
    num_vis = args.num_vis
    vis_fname= args.vis_fname
    is_ddp_trained = args.ddp_trained

    if is_ddp_trained:
        print("Testing DDP-trained model...")
        ddp=True
    else:
        print("Testing non-DDP-trained model...")
        ddp=False
    
    diffusion_model, config_dict = build_comp_diff_from_cfg(test_model, ddp=ddp)
    
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
    
    # Run tests
    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(
        policy, config_dict=config_dict, num_tests=test_num, map_size=map_size, num_vis=num_vis, device=device, vis_fname=vis_fname)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    results = main()