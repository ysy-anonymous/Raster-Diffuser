import numpy as np
from utils.value_utils import create_test_scenario, show_multiple_with_collision_colors
from utils.load_utils import build_rediff_from_cfg
# Diffusion Policy for testing samples...
from core.rediffuser.networks.diffuser.guides.policies import Policy
import torch
import argparse

# norm_stats: Normalization stats
def generate_path(policy, start, goal, obstacles, norm_stats):
    device = start.device

    start_np = start.cpu().numpy()[0]
    goal_np = goal.cpu().numpy()[0]
    env_cond = np.concatenate([start_np, goal_np])  # [2 * obs_dim]
    map_cond = obstacles[0].cpu().numpy()  # shape: [1, 8, 8]

    trajectory = policy(map_cond=map_cond, st_gl=env_cond, device=device, stats=norm_stats, dataset_mode=False)

    return torch.tensor(trajectory, device=device) # [1, H, obs_dim]

def parse_argument():
    parser = argparse.ArgumentParser(description='test diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: ReDiffuser Default')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model to test is trained with DDP')
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
        # try:
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
        trajectory = generate_path(policy, start, goal, obstacles, config_dict['normalizer'])
        
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
            
        # except Exception as e:
            # print(f"Error in test {i+1}: {str(e)}")
            # continue
    
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
    test_num = args.test_num
    map_size = args.map_size
    num_vis = args.num_vis
    vis_fname= args.vis_fname

    if is_ddp_trained:
        ddp=True
    else:
        ddp=False
    
    diffusion_model, config_dict = build_rediff_from_cfg(test_model, ddp=ddp)
    
    policy = Policy(
        diffusion_model=diffusion_model,
        config=config_dict,
        horizon=config_dict['diffusion_config']['horizon']
    )
    
    # Load weights
    policy.load_weights(ckpt_path)
    print("Model loaded successfully!")
    
    # Run tests
    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(
        policy, num_tests=test_num, device=device, config_dict=config_dict, map_size=map_size, num_vis=num_vis, vis_fname=vis_fname)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    results = main()