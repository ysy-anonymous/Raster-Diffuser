import numpy as np
from utils.value_utils import create_test_scenario, generate_path, show_multiple_with_collision_colors
from core.diffuser.diffusion.diffusion import PlaneDiffusionPolicy
from utils.load_utils import build_noise_scheduler_from_config, build_networks_from_config
import torch
import argparse
import random

def parse_argument():
    parser = argparse.ArgumentParser(description='test diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: default model, 1: transformer decoder, 2: default + transformer, ' \
    '3: diffusion planner, 4: bilatent diffusion planner, 5: multi-hypothesis diffusion planner, 6: trajectory planner plus, 7: multi resolution timestep trajectory planner, \
    8: trajectory planner 2nd gen')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model was trained with DDP (affects how weights are loaded)')
    parser.add_argument('--cp_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/conv/ckpt_final.ckpt', help='trained model weight path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run model')
    parser.add_argument('--test_num', type=int, default=100, help='number of test scenarios to generate')
    parser.add_argument('--map_size', type=int, default=8, help='size of the map (map_size x map_size)')
    parser.add_argument('--num_vis', type=int, default=100, help='number of test scenarios to visualize (if test_num > num_vis, then randomly select num_vis samples for visualization)')
    parser.add_argument('--vis_fname', type=str, default='test_results', help='name of visualization file')
    parser.add_argument('--test_seed', type=int, default=0, help='set manual number in torch.manual_seed()')
    parser.add_argument('--data_seed', type=int, default=0, help='set data generation RNG seed number')
    args = parser.parse_args()
    
    return args


def test_diffusion_policy(policy, config_dict, num_tests=20, map_size=8, num_vis=100, device='cuda', vis_fname='test_results', data_seed=0):
    """
    Test diffusion policy using value_utils functions
    """

    # Test parameters
    bounds = np.array([[0.0, map_size], [0.0, map_size]])
    cell_size = 1.0
    origin = bounds[:, 0]
    max_rectangles = (3 * (map_size//8), 5 * (map_size//8))
    rng = np.random.default_rng(data_seed) # random index...
    
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
        initial_action = torch.randn((1, config_dict["horizon"], config_dict["action_dim"]), device=device)
        trajectory, _ = generate_path(policy, start, goal, obstacles, initial_action) # use this when you normalize start/goal point
        
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
    ckpt_path = args.cp_path
    device = args.device
    test_model = args.model_id
    test_num = args.test_num
    map_size = args.map_size
    num_vis = args.num_vis
    vis_fname = args.vis_fname
    is_ddp_trained = args.ddp_trained
    test_seed = args.test_seed
    data_seed = args.data_seed

    if is_ddp_trained:
        print("Model was trained with DDP. Make sure to load weights accordingly.")
        ddp=True
    else:
        ddp=False

    print("Loading diffusion policy model...")
    config, config_dict, net = build_networks_from_config(test_model, ddp=ddp)
    scheduler = build_noise_scheduler_from_config(config_dict)
    
    policy = PlaneDiffusionPolicy(
        model=net, 
        noise_scheduler=scheduler, 
        config=config_dict, 
        device=device
    )
    
    # Load weights
    policy.load_weights(ckpt_path)
    print("Model loaded successfully!")
    
    torch.manual_seed(test_seed) # set test seed

    # Run tests
    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(
        policy, num_tests=test_num, device=device, config_dict=config_dict, map_size=map_size, num_vis=num_vis, vis_fname=vis_fname, data_seed=data_seed)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    results = main()