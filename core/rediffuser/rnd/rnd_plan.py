import numpy as np
import os
import torch
import argparse

from rnd import RNDModel
from utils.value_utils import create_test_scenario, show_multiple_with_collision_colors
from utils.load_utils import build_rediff_from_cfg
from core.rediffuser.networks.diffuser.guides.policies import Policy

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=int, default=0, help='0: ReDiffuser')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model to test is trained with DDP')
    parser.add_argument('--cp_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt', help='model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to inference')
    parser.add_argument('--test_num', type=int, default=1000, help='number of test samples to generate')
    parser.add_argument('--map_size', type=int, default=8, help='test set map size')
    parser.add_argument('--num_vis', type=int, default=100, help='test samples for visualization')
    parser.add_argument('--vis_fname', type=str, default='test_results', help='name of visualization file')

    parser.add_argument('--n_plans', type=int, default=8, help='number of plans to generate for each test scenario')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--rnd_keyword', type=str, default='rrt_8x8_100k', help='keyword for RND dataset to use for robust planning')
    parser.add_argument('--rnd_n_epochs', type=int, default=100, help='number of epochs for the RND model to load (used for constructing the load path)')
    parser.add_argument("--prob_sample", action="store_true", default=False, help="sample plan with softmax")
    parser.add_argument("--discount_power", type=float, default=0.0, help="how much we doubt at shorter horizons")

    args = parser.parse_args()
    return args

def soft_max(a):
    
    assert len(a.shape) == 1
    a -= np.max(a)
    
    return np.exp(a) / np.sum(np.exp(a))


# (n, discount_H, 4)
def fit_model(n_states, horizon, pad=False):
    
    n_states_shape = n_states.shape
    
    # move target state to the first position
    n_fit_states = np.concatenate((n_states[:, -1:, :], n_states[:, :-1, :]), axis=1)
    
    if pad:
        n_fit_states = np.concatenate(
            (n_fit_states, np.tile(n_fit_states[:, :1, :], (1, int(horizon-n_states_shape[1]), 1))),
            axis=1
        )
    
    ### 3. fit data to robuster: (n, H, 2)
    # rnd: (n, 2, H)
    n_fit_states = np.transpose(n_fit_states, (0, 2, 1))    
    return n_fit_states


def revert4plan(fit_states):
        
    ### 4. revert data for planning: --> (H, 2)
    
    # rnd: (2, H)
    reverted_states = np.transpose(fit_states, axes=(1, 0))

    # move target state to the last position
    reverted_states = np.concatenate((reverted_states[1:], reverted_states[:1]))
    
    return reverted_states


# shape is fit to robuster, e.g. rnd: (n, 2, H)
def choose_trajectory(n_fit_states, model, device, do_prob_sample, discount=1.0):
    
    inputs = torch.FloatTensor(n_fit_states).to(device)
    
    ### 5. convert model output to uncertainty
    # rnd
    n_uncertainty = model(inputs).detach().cpu().data.numpy().flatten() / discount
    
    if do_prob_sample:
        fit_states = None
        
        probs = soft_max(-n_uncertainty)
        
        sample_prob = np.random.random()
        cum_prob = 0.
        
        for (prob, fit_states) in zip(probs, n_fit_states):
            cum_prob += prob
            if sample_prob < cum_prob:
                break
        return fit_states, n_uncertainty.mean()
    
    else:
        min_idx = np.argmin(n_uncertainty)
        return n_fit_states[min_idx], n_uncertainty.mean()
    

# norm_stats: Normalization stats
def generate_path(policy, rnd_model, start, goal, obstacles, norm_stats, planning_cfg):
    device = start.device

    start_np = start.cpu().numpy()[0]
    goal_np = goal.cpu().numpy()[0]
    env_cond = np.concatenate([start_np, goal_np])  # [2 * obs_dim]
    map_cond = obstacles[0].cpu().numpy()  # shape: [1, 8, 8]

    for plan_id in range(planning_cfg["n_plans"]):

        best_states = None
        lowest_uncertainty = np.inf
        best_horizon = 0

        trajectory = policy(map_cond=map_cond, st_gl=env_cond, device=device, stats=norm_stats, dataset_mode=False)
        trajectory = trajectory.cpu().numpy() # convert to numpy for fitting and RND evaluation

        n_fit_states = fit_model(trajectory, horizon=policy.horizon, pad=(policy.horizon < planning_cfg["H"]))
        fit_states, uncertainty = choose_trajectory(n_fit_states, rnd_model, device=device, do_prob_sample=planning_cfg['prob_sample'],
                                                     discount=(policy.horizon/planning_cfg["H"])**planning_cfg['discount_power'])
        
        if uncertainty < lowest_uncertainty:
            best_states = revert4plan(fit_states)
            best_horizon = policy.horizon
            lowest_uncertainty = uncertainty

        # print("{}: Best horizon has been chosen with {}".format(plan_id + 1, best_horizon))
        plan_states = best_states      

    plan_states = np.array(plan_states)
    # print("plan_states shape: ", plan_states.shape)

    # Ouput as numpy array
    return plan_states # [1, H, obs_dim]


# Test Trained Models
def test_diffusion_policy(policy, rnd_model, config_dict, planning_cfg, num_tests=20, map_size=8, num_vis=100, device='cuda', vis_fname='test_results'):
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
        trajectory = generate_path(policy, rnd_model, start, goal, obstacles, config_dict['normalizer'], planning_cfg)
        
        # Convert to numpy for visualization
        grid_np = obstacles[0, 0].cpu().numpy()
        path_np = trajectory # here trajectory is already in numpy format and does no have batch dimension
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
    test_model = args.model_id
    is_ddp_trained = args.ddp_trained
    ckpt_path = args.cp_path
    device = args.device
    test_num = args.test_num
    map_size = args.map_size
    num_vis = args.num_vis
    vis_fname= args.vis_fname

    seed = args.seed
    rnd_keyword = args.rnd_keyword
    rnd_n_epochs = args.rnd_n_epochs
    n_plans = args.n_plans # number of plans to generate for each test scenario
    if args.prob_sample:
        prob_sample=True
    else:
        prob_sample=False
    discount_power = args.discount_power

    # Set seed
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ddp=True if is_ddp_trained else False

    robuster = 'rnd'
    base_load_dir = '/exhdd/seungyu/diffusion_motion/core/rediffuser/rnd'
    rnd_load_dir = os.path.join(base_load_dir, "RNDModel/model_{}_{}_{}e".format(robuster, rnd_keyword, rnd_n_epochs))

    # Setting RND Model for robust planning
    if rnd_keyword == "rrt_8x8_100k":
        H = 32
        output_dim = 32
    else:
        raise NotImplementedError("Keyword {} is not currently supported".format(args.keyword))
    # rnd
    input_size = (2, H)
    rnd_model = RNDModel(input_size=input_size, output_dim=output_dim).to(device)
    rnd_model.target.load_state_dict(torch.load(os.path.join(rnd_load_dir, "target.pth")))
    rnd_model.predictor.load_state_dict(torch.load(os.path.join(rnd_load_dir, "predictor.pth")))
    rnd_model.eval()
    print("RND Model Loaded Succesffuly")

    # Setting Diffusion Models & Policy
    diffusion_model, config_dict = build_rediff_from_cfg(test_model, ddp=ddp)
    diffusion_model = diffusion_model.to(device)
    policy = Policy(
        diffusion_model=diffusion_model,
        config=config_dict,
        horizon=config_dict['diffusion_config']['horizon']
    )
    policy.load_weights(ckpt_path)
    print("Diffusion Model loaded successfully!")

    planning_cfg = {
        "n_plans": n_plans,
        "prob_sample": prob_sample,
        "discount_power": discount_power,
        "H": H # planning horizon for uncertainty estimation by RND Model.
    }

    # Run tests
    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(
        policy, rnd_model, num_tests=test_num, device=device, config_dict=config_dict, planning_cfg=planning_cfg,
             map_size=map_size, num_vis=num_vis, vis_fname=vis_fname)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    main()