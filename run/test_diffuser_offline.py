from core.diffuser.diffusion.diffusion import PlaneDiffusionPolicy
from core.diffuser.datasets.plane_dataset_embeed import PlanePlanningDataSets
from utils.value_utils import show_multiple_with_collision_colors
from utils.load_utils import build_noise_scheduler_from_config, build_networks_from_config
import numpy as np
import torch
import argparse
import random

def to_index_list(x):
    if x is None:
        return []

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.size == 0:
        return []

    return x.reshape(-1).astype(int).tolist()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_argument():
    parser = argparse.ArgumentParser(description='test diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: default model, 1: transformer decoder, 2: default + transformer, ' \
    '3: diffusion planner, 4: bilatent diffusion planner, 5: multi-hypothesis diffusion planner, 6: trajectory planner plus, 7: multi resolution timestep trajectory planner, 8: trajectory planner 2nd gen')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model was trained with DDP (affects how weights are loaded)')
    parser.add_argument('--cp_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/conv/ckpt_final.ckpt', help='trained model weight path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run model')
    parser.add_argument('--num_vis', type=int, default=500, help='number of test scenarios to visualize (if test_num > num_vis, then randomly select num_vis samples for visualization)')
    parser.add_argument('--vis_fname', type=str, default='test_results', help='name of visualization file')
    parser.add_argument('--test_path', type=str, default='datasets/test_scenarios_25000_8x8.npy', help='path to test scenarios npy file')
    parser.add_argument('--test_bs', type=int, default=2048, help='batch size for testing')
    parser.add_argument('--strict_seed', action='store_true', help='if set, will use the specific random seed for all randomness in the test')
    parser.add_argument('--seed', type=int, default=0, help='random seed for controlling diffusion stochastic sampling and randomness in torch operations')
    parser.add_argument('--test_failure_case', action='store_true', help='if set, will save failure cases for later analysis')

    args = parser.parse_args()    
    return args


def make_failure_dataloader(
        test_dataset,
        failure_cases,
        batch_size,
        num_workers=0,
        pin_memory=False,
        collate_fn=None,
):
    """
    failure_cases example:
    {
        "collision_case": [0, 3, 10],
        "failed_to_reach": [5, 10, 12],
    }
    """

    collision_indices = to_index_list(failure_cases.get("collision_case", []))
    failed_reach_indices = to_index_list(failure_cases.get("failed_to_reach", []))

    # Merge, remove duplicates, and sort for reproducibility
    # failure_indices = sorted(set(collision_indices + failed_reach_indices))/
    failure_indices = sorted(set(collision_indices).union(failed_reach_indices))

    print(f"Number of unique failure cases: {len(failure_indices)}")
    # print(f"Failure indices: {failure_indices}")

    failure_subset = torch.utils.data.Subset(test_dataset, failure_indices)
    failure_dataloader = torch.utils.data.DataLoader(
        failure_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return failure_dataloader, failure_indices



def generate_batched_path(policy, start, goal, obstacles, initial_action, bs=None, generator=None):
    device = start.device
    obs_dim = start.shape[1]
    B = start.shape[0]
    pred_horizon = policy.config["horizon"]

    fake_obs_sample = np.zeros((B, pred_horizon, obs_dim), dtype=np.float32)

    start_np = start.cpu().numpy()
    goal_np = goal.cpu().numpy()

    env_cond = np.concatenate([start_np, goal_np], axis=-1)
    map_cond = obstacles.cpu().numpy()

    obs_dict = {
        "sample": fake_obs_sample,
        "env": env_cond,
        "map": map_cond,
    }

    if initial_action.shape[0] != B:
        initial_action = initial_action[:B]

    trajectory, trajectory_all = policy.predict_action(
        obs_dict,
        initial_action,
        generator=generator,
    )

    return torch.tensor(trajectory, device=device), trajectory_all


def test_diffusion_policy(
    policy,
    test_dataloader,
    config_dict,
    num_vis=100,
    device="cuda",
    vis_fname="test_results",
    bs=1,
    seed=None,
    test_failure_mode=False,
    dataset_indices=None,
    per_case_seed=True,
):
    """
    Evaluate diffusion policy.

    Args:
        policy:
            PlaneDiffusionPolicy.

        test_dataloader:
            Dataloader for either full test dataset or failure-case subset.

        config_dict:
            Model/diffusion config.

        num_vis:
            Number of samples to visualize.

        device:
            CUDA / CPU device.

        vis_fname:
            Output visualization filename.

        bs:
            Nominal dataloader batch size.
            This is used only for compatibility. The function uses actual batch size B internally.

        seed:
            Base seed. seed=0 is valid.
            If None, randomness is uncontrolled.

        test_failure_mode:
            Whether this is failure-case re-evaluation mode.

        dataset_indices:
            Original dataset indices corresponding to the dataloader.
            For full test dataloader, can be None.
            For failure subset, pass failure_indices.

        per_case_seed:
            If True, each original scenario gets its own generator:
                case_seed = seed * 1_000_003 + original_idx
            This makes results independent of batch size and subset ordering.

            If False, one generator is shared across the whole dataloader.
            This is faster, but changing batch size/subset changes RNG assignment.
    """

    grid_list = []
    path_list = []
    start_list = []
    goal_list = []
    original_idx_list = []

    policy.net.eval()

    if dataset_indices is not None:
        dataset_indices = [int(x) for x in dataset_indices]

    print(f"Testing {len(test_dataloader)} batches...")
    if seed is not None:
        print(f"Base seed: {seed}")
        print(f"Per-case seed mode: {per_case_seed}")

    # Used only when per_case_seed=False
    if seed is not None and not per_case_seed:
        shared_gen = torch.Generator(device=device).manual_seed(int(seed))
    else:
        shared_gen = None

    cursor = 0

    for batch_idx, data in enumerate(test_dataloader):
        map_cond = data["map"].to(device)      # [B, C, H, W]
        env_cond = data["env"].to(device)      # [B, 2 * obs_dim]

        start = env_cond[:, :2]                # [B, 2]
        goal = env_cond[:, 2:]                 # [B, 2]

        B = start.shape[0]

        # Recover original dataset indices for this batch.
        # This is important when test_dataloader is a Subset dataloader.
        if dataset_indices is None:
            batch_original_indices = list(range(cursor, cursor + B))
        else:
            batch_original_indices = dataset_indices[cursor : cursor + B]

        cursor += B

        if per_case_seed:
            # Slower but reproducible per original scenario.
            # Each sample is generated independently with its own seed.
            batch_trajectories = []

            for local_j, original_idx in enumerate(batch_original_indices):
                single_start = start[local_j : local_j + 1]       # [1, 2]
                single_goal = goal[local_j : local_j + 1]         # [1, 2]
                single_map = map_cond[local_j : local_j + 1]      # [1, C, H, W]

                if seed is not None:
                    case_seed = int(seed) * 1_000_003 + int(original_idx)
                    gen = torch.Generator(device=device).manual_seed(case_seed)
                else:
                    gen = None

                initial_action = torch.randn(
                    (
                        1,
                        config_dict["horizon"],
                        config_dict["action_dim"],
                    ),
                    device=device,
                    generator=gen,
                )

                with torch.no_grad():
                    trajectory, _ = generate_batched_path(
                        policy=policy,
                        start=single_start,
                        goal=single_goal,
                        obstacles=single_map,
                        initial_action=initial_action,
                        bs=1,
                        generator=gen,
                    )

                batch_trajectories.append(trajectory)

            trajectory = torch.cat(batch_trajectories, dim=0)     # [B, H, action_dim]

        else:
            # Faster batched mode.
            # Reproducible only if dataloader order and batch size are fixed.
            initial_action = torch.randn(
                (
                    B,
                    config_dict["horizon"],
                    config_dict["action_dim"],
                ),
                device=device,
                generator=shared_gen,
            )

            with torch.no_grad():
                trajectory, _ = generate_batched_path(
                    policy=policy,
                    start=start,
                    goal=goal,
                    obstacles=map_cond,
                    initial_action=initial_action,
                    bs=B,
                    generator=shared_gen,
                )

        grid_list.extend(map_cond[:, 0].detach().cpu().numpy())
        path_list.extend(trajectory.detach().cpu().numpy())
        start_list.extend(start.detach().cpu().numpy())
        goal_list.extend(goal.detach().cpu().numpy())
        original_idx_list.extend(batch_original_indices)

        print(
            f"Batch {batch_idx + 1}/{len(test_dataloader)} completed "
            f"with B={B}"
        )

    print(f"Generated {len(path_list)} valid test cases")

    local_indices = list(range(len(path_list)))

    # If your show_multiple_with_collision_colors() supports original_indices,
    # use this version.
    try:
        results = show_multiple_with_collision_colors(
            grid_list,
            path_list,
            start_list,
            goal_list,
            local_indices,
            cols=5,
            num_vis=num_vis,
            vis_fname=vis_fname,
            test_failure_mode=test_failure_mode,
            seed=seed,
            original_indices=original_idx_list,
        )

    # If your current visualization function does not yet accept original_indices,
    # this fallback keeps the evaluation working.
    except TypeError:
        results = show_multiple_with_collision_colors(
            grid_list,
            path_list,
            start_list,
            goal_list,
            local_indices,
            cols=5,
            num_vis=num_vis,
            vis_fname=vis_fname,
            test_failure_mode=test_failure_mode,
            seed=seed,
        )

    results["original_indices"] = original_idx_list
    results["seed"] = seed
    results["per_case_seed"] = per_case_seed

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
    strict = args.strict_seed
    seed = args.seed
    test_failure = args.test_failure_case

    if strict:
        print(f"Using strict random seed: {seed} for all operations.")
        set_global_seed(seed)
    
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

    # in this case, distance map calculation happens inside the model prediction loop. Do not set include_distance_map=True here.
    test_dataset = PlanePlanningDataSets(dataset_path=test_path, include_distance_map=False, cell_size=1.0, boundary_zero=False, input_upsample=1.0, normalize_st=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)

    if test_failure:
        failure_cases = np.load('/exhdd/seungyu/diffusion_motion/dataset/failure_case_16x16_raster_diffuser_seed0.npz', allow_pickle=True) if test_failure else None
        failure_dataloader, failure_indices = make_failure_dataloader(
            test_dataset,
            failure_cases=failure_cases,
            batch_size=test_bs,
        )

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

    # Run tests
    print("\nStarting diffusion policy evaluation...")
    if test_failure:
        # print(f"Testing on failure cases with indices: {failure_indices}")
        results = test_diffusion_policy(
            policy,
            test_dataloader=failure_dataloader,
            config_dict=config_dict,
            num_vis=num_vis,
            device=device,
            vis_fname=vis_fname,
            bs=test_bs,
            seed=seed,
            test_failure_mode=True,
            dataset_indices=failure_indices,
            per_case_seed=False
        )
    else:
        results = test_diffusion_policy(
            policy,
            test_dataloader=test_dataloader, 
            device=device,
            config_dict=config_dict, 
            num_vis=num_vis, 
            vis_fname=vis_fname, 
            bs=test_bs, 
            seed=seed, 
            test_failure_mode=False,
            dataset_indices=None,
            per_case_seed=False)
    
    print("\n=== Final Results ===")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Collision Rate: {results['collision_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    results = main()