import numpy as np
import os
import torch
import argparse

from rnd_v2 import ConditionalRNDModel
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
    parser.add_argument('--test_path', type=str, default='dataset/test_scenarios_2000.npy', help='path to test scenarios npy file')
    parser.add_argument('--test_bs', type=int, default=64, help='batch size for testing')

    parser.add_argument('--n_plans', type=int, default=3, help='number of replanning in iterative manner.')
    parser.add_argument('--n_alter', type=int, default=5, help='number of alternative trajectories for stochasticity of diffusion model.')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')

    parser.add_argument('--rnd_keyword', type=str, default='rrt_8x8_100k', help='keyword for RND dataset to use for robust planning')
    parser.add_argument('--rnd_n_epochs', type=int, default=100, help='number of epochs for the RND model to load')
    parser.add_argument("--prob_sample", action="store_true", default=False, help="sample plan with softmax")
    parser.add_argument("--discount_power", type=float, default=0.0, help="how much we doubt at shorter horizons")

    # Conditional RND settings
    parser.add_argument('--rnd_output_dim', type=int, default=256)
    parser.add_argument('--rnd_traj_feat_dim', type=int, default=128)
    parser.add_argument('--rnd_mask_feat_dim', type=int, default=128)
    parser.add_argument('--rnd_ctx_feat_dim', type=int, default=64)
    parser.add_argument('--rnd_context_dim', type=int, default=4)
    parser.add_argument('--rnd_mask_channels', type=int, default=3)

    args = parser.parse_args()
    return args


def unnormalize(data, stats):
    eps = 1e-8
    stats_np = {k: np.array(v) for k, v in stats.items()}
    denorm = (data + 1) / 2 * (stats_np['max'] - stats_np['min'] + eps) + stats_np['min']
    return denorm


def soft_max(a):
    a = a - np.max(a, axis=-1, keepdims=True)
    return np.exp(a) / np.sum(np.exp(a), axis=-1, keepdims=True)


# (n, discount_H, 2) -> (n, 2, H_fit)
def fit_model(n_states, horizon, pad=False):
    n_states_shape = n_states.shape

    # move target state to the first position
    n_fit_states = np.concatenate((n_states[:, -1:, :], n_states[:, :-1, :]), axis=1)

    if pad:
        n_fit_states = np.concatenate(
            (n_fit_states, np.tile(n_fit_states[:, :1, :], (1, int(horizon - n_states_shape[1]), 1))),
            axis=1
        )

    n_fit_states = np.transpose(n_fit_states, (0, 2, 1))
    return n_fit_states


def revert4plan(fit_states):
    # (B, 2, H) -> (B, H, 2)
    reverted_states = np.transpose(fit_states, axes=(0, 2, 1))
    # move target state back to the last position
    reverted_states = np.concatenate((reverted_states[:, 1:], reverted_states[:, :1]), axis=1)
    return reverted_states


def repeat_conditions_for_alternatives(map_cond, env_cond, batch_repeat):
    """
    map_cond: (B, C, Hm, Wm) or (B, Hm, Wm)
    env_cond: (B, D)
    returns repeated arrays for candidate trajectories: (B*batch_repeat, ...)
    """
    if map_cond.ndim == 3:
        map_cond = np.expand_dims(map_cond, axis=1)

    map_rep = np.repeat(map_cond, batch_repeat, axis=0)
    env_rep = np.repeat(env_cond, batch_repeat, axis=0)
    return map_rep, env_rep


def choose_trajectory_conditional(
    n_fit_states,
    map_cond,
    env_cond,
    model,
    device,
    do_prob_sample,
    B,
    batch_repeat,
    horizon,
    obs_dim,
    discount=1.0
):
    """
    n_fit_states: (B*batch_repeat, obs_dim, horizon)
    map_cond:     (B, C, Hm, Wm) or (B, Hm, Wm)
    env_cond:     (B, D)
    """
    map_rep, env_rep = repeat_conditions_for_alternatives(map_cond, env_cond, batch_repeat)

    traj_tensor = torch.FloatTensor(n_fit_states).to(device)
    map_tensor = torch.FloatTensor(map_rep).to(device)
    env_tensor = torch.FloatTensor(env_rep).to(device)

    with torch.no_grad():
        n_uncertainty = model(traj_tensor, map_tensor, env_tensor).detach().cpu().numpy().flatten() / discount

    if do_prob_sample:
        n_uncertainty = n_uncertainty.reshape(B, batch_repeat)
        probs = soft_max(-n_uncertainty)

        n_fit_states = n_fit_states.reshape(B, batch_repeat, obs_dim, horizon)

        batch_fit_states = []
        batch_uncertainty = []

        for b in range(B):
            sampled_idx = np.random.choice(batch_repeat, p=probs[b])
            batch_fit_states.append(n_fit_states[b, sampled_idx])
            batch_uncertainty.append(n_uncertainty[b].mean())

        batch_fit_states = np.stack(batch_fit_states, axis=0)
        batch_uncertainty = np.array(batch_uncertainty)

        return batch_fit_states, batch_uncertainty

    else:
        n_fit_states = n_fit_states.reshape(B, batch_repeat, obs_dim, horizon)
        n_uncertainty = n_uncertainty.reshape(B, batch_repeat)

        min_idx = np.argmin(n_uncertainty, axis=-1)
        selected_fit_states = n_fit_states[np.arange(B), min_idx]
        mean_uncertainty = n_uncertainty.mean(axis=-1)

        return selected_fit_states, mean_uncertainty


def generate_path(policy, rnd_model, start, goal, obstacles, norm_stats, planning_cfg):
    device = start.device

    start_np = start.detach().cpu().numpy()
    goal_np = goal.detach().cpu().numpy()
    env_cond = np.concatenate([start_np, goal_np], axis=-1)   # (B, 4)
    map_cond = obstacles.detach().cpu().numpy()               # (B, C, Hm, Wm) or similar

    best_states = None
    lowest_uncertainty = np.full(env_cond.shape[0], np.inf)
    best_horizon = np.zeros(env_cond.shape[0], dtype=int)

    for plan_id in range(planning_cfg["n_plans"]):
        trajectory = policy(
            map_cond=map_cond,
            st_gl=env_cond,
            device=device,
            stats=norm_stats,
            batch_repeat=planning_cfg['n_alter'],
            dataset_mode=True
        )

        trajectory = trajectory.detach().cpu().numpy()  # (B*n_alter, H, 2)

        n_fit_states = fit_model(
            trajectory,
            horizon=policy.horizon,
            pad=(policy.horizon < planning_cfg["H"])
        )

        fit_states, uncertainty = choose_trajectory_conditional(
            n_fit_states=n_fit_states,
            map_cond=map_cond,
            env_cond=env_cond,
            model=rnd_model,
            device=device,
            do_prob_sample=planning_cfg['prob_sample'],
            B=env_cond.shape[0],
            batch_repeat=planning_cfg['n_alter'],
            horizon=planning_cfg["H"],
            obs_dim=2,
            discount=(policy.horizon / planning_cfg["H"]) ** planning_cfg['discount_power']
        )

        candidate_states = revert4plan(fit_states)
        mask = uncertainty < lowest_uncertainty

        if best_states is None:
            best_states = candidate_states.copy()
        else:
            best_states[mask] = candidate_states[mask]

        lowest_uncertainty[mask] = uncertainty[mask]
        best_horizon[mask] = policy.horizon

    plan_states = np.array(best_states)
    return plan_states


def test_diffusion_policy(policy, test_dataloader, rnd_model, config_dict, planning_cfg, num_vis=100, device='cuda', vis_fname='test_results'):
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

        trajectory = generate_path(
            policy, rnd_model, start, goal, map_cond,
            config_dict['normalizer'], planning_cfg
        )

        grid_np = map_cond[:, 0].detach().cpu().numpy()
        path_np = trajectory
        start_np = start.detach().cpu().numpy()
        goal_np = goal.detach().cpu().numpy()

        grid_list.extend(grid_np)
        path_list.extend(path_np)
        start_list.extend(unnormalize(start_np, config_dict['normalizer']))
        goal_list.extend(unnormalize(goal_np, config_dict['normalizer']))

        print(f"Test {i+1}/{len(test_dataloader)} completed")

    print(f"Generated {len(path_list)} valid test cases")

    results = show_multiple_with_collision_colors(
        grid_list, path_list, start_list, goal_list,
        list(range(len(path_list))), cols=5, num_vis=num_vis, vis_fname=vis_fname
    )

    return results


def main():
    args = parse_argument()

    test_model = args.model_id
    is_ddp_trained = args.ddp_trained
    ckpt_path = args.cp_path
    device = args.device
    num_vis = args.num_vis
    vis_fname = args.vis_fname
    test_path = args.test_path
    test_bs = args.test_bs

    seed = args.seed
    rnd_keyword = args.rnd_keyword
    rnd_n_epochs = args.rnd_n_epochs
    n_plans = args.n_plans
    n_alter = args.n_alter
    prob_sample = bool(args.prob_sample)
    discount_power = args.discount_power

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ddp = True if is_ddp_trained else False

    robuster = 'rnd'
    base_load_dir = '/exhdd/seungyu/diffusion_motion/core/rediffuser/rnd'
    rnd_load_dir = os.path.join(
        base_load_dir,
        "RNDModelV2",
        "model_{}_{}_{}e".format(robuster, rnd_keyword, rnd_n_epochs)
    )

    if rnd_keyword in ["rrt_8x8_100k", "rrt_8x8_40k", "rrt_8x8_2k"]:
        H = 32
    elif rnd_keyword in ['rrt_16x16_100k']:
        H = 64
    else:
        raise NotImplementedError(f"Keyword {rnd_keyword} is not currently supported")

    rnd_model = ConditionalRNDModel(
        traj_channels=2,
        rnd_output_dim=args.rnd_output_dim,
        context_dim=args.rnd_context_dim,
        mask_in_channels=args.rnd_mask_channels,
        traj_feat_dim=args.rnd_traj_feat_dim,
        mask_feat_dim=args.rnd_mask_feat_dim,
        ctx_feat_dim=args.rnd_ctx_feat_dim, 
    ).to(device)

    # Prefer full model state if available
    full_model_path = os.path.join(rnd_load_dir, "conditional_rnd_model.pth")
    predictor_only_path = os.path.join(rnd_load_dir, "predictor_only.pth")

    if os.path.exists(full_model_path):
        state_dict = torch.load(full_model_path, map_location=device)
        rnd_model.load_state_dict(state_dict, strict=True)
        print("Conditional RND full model loaded successfully")
    elif os.path.exists(predictor_only_path):
        current_state = rnd_model.state_dict()
        loaded_state = torch.load(predictor_only_path, map_location=device)
        current_state.update(loaded_state)
        rnd_model.load_state_dict(current_state, strict=False)
        print("Conditional RND predictor-only weights loaded successfully")
    else:
        raise FileNotFoundError(
            f"Could not find Conditional RND checkpoint in {rnd_load_dir}"
        )

    rnd_model.eval()

    diffusion_model, config_dict = build_rediff_from_cfg(test_model, ddp=ddp)
    diffusion_model = diffusion_model.to(device)

    policy = Policy(
        diffusion_model=diffusion_model,
        config=config_dict,
        horizon=config_dict['diffusion_config']['horizon']
    )
    policy.load_weights(ckpt_path)
    print("Diffusion model loaded successfully!")

    test_dataset = PlanePlanningDataSets(dataset_path=test_path, **config_dict['dataset'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)

    # # here is subset
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
        "H": H
    }

    print("\nStarting diffusion policy evaluation...")
    results = test_diffusion_policy(
        policy,
        test_dataloader=test_dataloader,
        rnd_model=rnd_model,
        device=device,
        config_dict=config_dict,
        planning_cfg=planning_cfg,
        num_vis=num_vis,
        vis_fname=vis_fname
    )

    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")

    return results


if __name__ == "__main__":
    main()