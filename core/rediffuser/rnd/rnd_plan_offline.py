import numpy as np
import os
import torch
import argparse

from rnd import RNDModel
from utils.value_utils import create_test_scenario, show_multiple_with_collision_colors
from utils.load_utils import build_rediff_from_cfg
from core.rediffuser.networks.diffuser.guides.policies import Policy
from core.rediffuser.datasets.plane_dataset_embeed import PlanePlanningDataSets

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=int, default=0, help='0: ReDiffuser')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model to test is trained with DDP')
    parser.add_argument('--cp_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt', help='model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to inference')
    parser.add_argument('--num_vis', type=int, default=100, help='test samples for visualization')
    parser.add_argument('--vis_fname', type=str, default='test_results', help='name of visualization file')
    parser.add_argument('--test_path', type=str, default='dataset/test_scenarios_2000.npy', help ='path to test scenarios npy file')
    parser.add_argument('--test_bs', type=int, default=64, help='batch size for testing')

    parser.add_argument('--n_plans', type=int, default=3, help='number of replanning in iterative manner.')
    parser.add_argument('--n_alter', type=int, default=5, help='number of alternative trajectories for stochasticity of diffusion model.')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--rnd_keyword', type=str, default='rrt_8x8_100k', help='keyword for RND dataset to use for robust planning')
    parser.add_argument('--rnd_n_epochs', type=int, default=100, help='number of epochs for the RND model to load (used for constructing the load path)')
    parser.add_argument("--prob_sample", action="store_true", default=False, help="sample plan with softmax")
    parser.add_argument("--discount_power", type=float, default=0.0, help="how much we doubt at shorter horizons")

    args = parser.parse_args()
    return args

def unnormalize(data, stats):
    eps = 1e-8
    for k, v in stats.items():
        stats[k] = np.array(v)
    denorm = (data + 1) / 2 * (stats['max'] - stats['min'] + eps) + stats['min']  # → [min, max]
    return denorm

def soft_max(a):
    
    a -= np.max(a, axis=-1)
    
    return np.exp(a) / np.sum(np.exp(a), axis=-1)


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
    
    # rnd: (B, 2, H) -> (B, H, 2)
    reverted_states = np.transpose(fit_states, axes=(0, 2, 1))

    # move target state to the last position
    reverted_states = np.concatenate((reverted_states[:, 1:], reverted_states[:, :1]), axis=1)
    
    return reverted_states


# shape is fit to robuster, e.g. rnd: (n, 2, H)
def choose_trajectory(n_fit_states, model, device, do_prob_sample,
                       B, batch_repeat, horizon, obs_dim, discount=1.0):
    
    inputs = torch.FloatTensor(n_fit_states).to(device)
    
    ### 5. convert model output to uncertainty
    # rnd
    n_uncertainty = model(inputs).detach().cpu().data.numpy().flatten() / discount
    if do_prob_sample:
        assert False, "Currently not supported!"
        fit_states = None
        n_uncertainty = n_uncertainty.reshape(B, batch_repeat)

        probs = soft_max(-n_uncertainty)
        
        sample_prob = np.random.random()
        cum_prob = 0.
        n_fit_states = n_fit_states.reshape(B, batch_repeat, obs_dim, horizon)

        batch_fit_states = []
        for batch_repeated in n_fit_states:
            for (prob, fit_states) in zip(probs, batch_repeated):
                cum_prob += prob
                if sample_prob < cum_prob:
                    break
            batch_fit_states.append(fit_states)

        batch_fit_states = torch.tensor(batch_fit_states)
        n_uncertainty = n_uncertainty.mean(axis=-1) # average the n_uncertainty.

        return batch_fit_states, n_uncertainty.mean()
    
    else:
        n_fit_states = n_fit_states.reshape(B, batch_repeat, obs_dim, horizon)
        n_uncertainty = n_uncertainty.reshape(B, batch_repeat)
        min_idx = np.argmin(n_uncertainty, axis=-1) # compute minimal uncertainty following the batch_repeat axis
        n_uncertainty = n_uncertainty.mean(axis=-1) # average the n_uncertainty.
        return n_fit_states[np.arange(B), min_idx], n_uncertainty
    

# norm_stats: Normalization stats
def generate_path(policy, rnd_model, start, goal, obstacles, norm_stats, planning_cfg):
    device = start.device

    start_np = start.cpu().numpy()
    goal_np = goal.cpu().numpy()
    env_cond = np.concatenate([start_np, goal_np], axis=-1)
    map_cond = obstacles.cpu().numpy()

    best_states = None
    lowest_uncertainty = np.full(env_cond.shape[0], np.inf)
    best_horizon = np.zeros(env_cond.shape[0], dtype=int)

    for plan_id in range(planning_cfg["n_plans"]):

        trajectory = policy(map_cond=map_cond, st_gl=env_cond, device=device, stats=norm_stats, batch_repeat=planning_cfg['n_alter'], dataset_mode=True)
        trajectory = trajectory.cpu().numpy() # convert to numpy for fitting and RND evaluation
        
        n_fit_states = fit_model(trajectory, horizon=policy.horizon, pad=(policy.horizon < planning_cfg["H"]))
        fit_states, uncertainty = choose_trajectory(n_fit_states, rnd_model, device=device, do_prob_sample=planning_cfg['prob_sample'],
                                                    B=env_cond.shape[0], batch_repeat=planning_cfg['n_alter'], horizon=32, obs_dim=2, # currently supports horizon=32, obs_dim=2
                                                     discount=(policy.horizon/planning_cfg["H"])**planning_cfg['discount_power'])
        candidate_states = revert4plan(fit_states)
        mask = uncertainty < lowest_uncertainty

        if best_states is None:
            best_states = candidate_states[mask]
        else:
            best_states[mask] = candidate_states[mask]
        
        lowest_uncertainty[mask] = uncertainty[mask]
        best_horizon[mask] = policy.horizon

        # print("{}: Best horizon has been chosen with {}".format(plan_id + 1, best_horizon))
        plan_states = best_states

    plan_states = np.array(plan_states)

    # Ouput as numpy array
    return plan_states # [1, H, obs_dim]


# Test Trained Models
def test_diffusion_policy(policy, test_dataloader, rnd_model, config_dict, planning_cfg, num_vis=100, device='cuda', vis_fname='test_results'):
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
        map_cond = data['map'].to(device)
        env_cond = data['env'].to(device)
        start = env_cond[:, :2]
        goal = env_cond[:, 2:]
        
        # Generate path using diffusion policy
        trajectory = generate_path(policy, rnd_model, start, goal, map_cond, config_dict['normalizer'], planning_cfg)

        # Convert to numpy for visualization
        grid_np = map_cond[:, 0].cpu().numpy()
        path_np = trajectory # here trajectory is already in numpy format and does no have batch dimension
        start_np = start.cpu().numpy()
        goal_np = goal.cpu().numpy()
        
        # Store results
        grid_list.extend(grid_np)
        path_list.extend(path_np)
        start_list.extend(unnormalize(start_np, config_dict['normalizer']))
        goal_list.extend(unnormalize(goal_np, config_dict['normalizer']))
        
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
    vis_fname= args.vis_fname
    test_path = args.test_path
    test_bs = args.test_bs

    seed = args.seed
    rnd_keyword = args.rnd_keyword
    rnd_n_epochs = args.rnd_n_epochs
    n_plans = args.n_plans # number of plans to generate for each test scenario
    n_alter = args.n_alter # number of alternative trajectories for each batch samples
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
    if rnd_keyword == "rrt_8x8_100k" or rnd_keyword == "rrt_8x8_40k" or rnd_keyword == 'rrt_8x8_2k':
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

    # in this case, distance map calculation happens inside the model prediction loop. Do not set include_distance_map=True here.
    test_dataset = PlanePlanningDataSets(dataset_path=test_path, **config_dict['dataset'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)

    # # here is subset for faster checking
    # sample_indices = []
    # dataset_len = len(test_dataset)
    # batch_size = test_bs
    # range_of_idx = range(0, 10, 1)
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

    planning_cfg = {
        "n_plans": n_plans,
        "n_alter": n_alter,
        "prob_sample": prob_sample,
        "discount_power": discount_power,
        "H": H # planning horizon for uncertainty estimation by RND Model.
    }

    # Run tests
    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(
        policy, test_dataloader=test_dataloader, rnd_model=rnd_model, device=device, config_dict=config_dict, planning_cfg=planning_cfg, num_vis=num_vis, vis_fname=vis_fname)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    main()