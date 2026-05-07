from core.pb_diffusion.diffusion.diffusion_policy import PBDiffusionPolicy
from core.pb_diffusion.datasets.plane_dataset_embeed import PlanePlanningDataSets

from utils.value_utils import show_multiple_with_collision_colors
from utils.load_utils import build_pb_diff_from_cfg
import numpy as np
import torch
import argparse
import random


def unnormalize(data, stats):
    eps = 1e-8
    for k, v in stats.items():
        stats[k] = np.array(v)
    denorm = (data + 1) / 2 * (stats['max'] - stats['min'] + eps) + stats['min']  # → [min, max]
    return denorm

def parse_argument():
    parser = argparse.ArgumentParser(description='test diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: Directly Adapted PBDMP, 1: PBDMP + BILatent Fusion (Not Yet Available), 2: V2 PBDMP')
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
def generate_path(policy: PBDiffusionPolicy, start, goal, obstacles):
    device = start.device
    obs_dim = start.shape[1]
    pred_horizon = policy.config["diffusion_config"]["horizon"]

    # Fake observation sample (all zeros)
    fake_obs_sample = np.zeros((pred_horizon, obs_dim), dtype=np.float32)

    start_np = start.cpu().numpy()
    goal_np = goal.cpu().numpy()
    env_cond = np.concatenate([start_np, goal_np], axis=-1)
    map_cond = obstacles.cpu().numpy()

    obs_dict = {
        "sample": fake_obs_sample,
        "env": env_cond,
        "map": map_cond,
    }

    trajectory, trajectory_all = policy.predict_action(obs_dict)  # shape: [pred_horizon, obs_dim]
    return torch.tensor(trajectory, device=device), trajectory_all  # [1, H, obs_dim]


def test_diffusion_policy(policy, test_dataloader, norm_stats, num_vis=100, device='cuda', vis_fname='test_results'):
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
        env_cond = data['env'].to(device) # (1, 2 * obs)
        start = env_cond[:, :2] # (1, 2)
        goal = env_cond[:, 2:] # (1, 2)
        
        # Generate path using diffusion policy
        trajectory, _ = generate_path(policy, start, goal, map_cond) # use this when you normalize start/goal point

        print("trajectory shape: ", trajectory.shape)

        # Convert to numpy for visualization
        grid_np = map_cond[:, 0].cpu().numpy() # [Bs, H, W]
        path_np = trajectory.cpu().numpy() # [Bs, 32, 2]
        start_np = unnormalize(start.cpu().numpy(), norm_stats) # [Bs, 2]
        goal_np = unnormalize(goal.cpu().numpy(), norm_stats) # [Bs, 2]

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
        print("Model was trained with DDP. Make sure to load weights accordingly.")
        ddp=True
    else:
        ddp=False

    print("Loading diffusion policy model...")
    diffusion_model, config_dict = build_pb_diff_from_cfg(test_model, ddp=ddp)

    policy = PBDiffusionPolicy(
        model=diffusion_model, 
        config=config_dict, 
        device=device
    )
    
    # Load weights
    policy.load_weights(ckpt_path)
    print("Model loaded successfully!")

    # in this case, distance map calculation happens inside the model prediction loop. Do not set include_distance_map=True here.
    test_dataset = PlanePlanningDataSets(dataset_path=test_path, include_distance_map=False, cell_size=1.0, boundary_zero=False, input_upsample=1.0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs,  shuffle=False)

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
    results = test_diffusion_policy(policy, test_dataloader=test_dataloader, norm_stats=config_dict['normalizer'], 
                                    device=device, num_vis=num_vis, vis_fname=vis_fname)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    results = main()