from core.diffuser.diffusion.diffusion import PlaneDiffusionPolicy
from core.diffuser.datasets.plane_dataset_embeed import PlanePlanningDataSets
from utils.load_utils import build_noise_scheduler_from_config, build_networks_from_config
import numpy as np
import torch
import argparse
import random
import time

def parse_argument():
    parser = argparse.ArgumentParser(description='test diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: default model, 1: transformer decoder, 2: default + transformer, ' \
    '3: diffusion planner, 4: bilatent diffusion planner, 5: multi-hypothesis diffusion planner, 6: trajectory planner plus, 7: multi resolution timestep trajectory planner, 8: trajectory planner 2nd gen')
    parser.add_argument('--ddp_trained', action='store_true', help='whether the model was trained with DDP (affects how weights are loaded)')
    parser.add_argument('--cp_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/conv/ckpt_final.ckpt', help='trained model weight path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run model')
    parser.add_argument('--test_path', type=str, default='datasets/test_scenarios_25000_8x8.npy', help='path to test scenarios npy file')
    parser.add_argument('--num_warmup', type=int, default=1000, help='number of warmup samples for speed test')
    parser.add_argument('--num_eval', type=int, default=1000, help='number of evaluation samples for speed test')

    args = parser.parse_args()    
    return args


# If norm_stats is provided, then normalize the start/goal points
def generate_batched_path(policy: PlaneDiffusionPolicy, start, goal, obstacles, initial_action, bs):
    device = start.device
    obs_dim = start.shape[1]
    B = start.shape[0]
    pred_horizon = policy.config["horizon"]

    # Fake observation sample (all zeros)
    fake_obs_sample = np.zeros((bs, pred_horizon, obs_dim), dtype=np.float32)

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
        print("remaining batch is different.. truncate initial_action batch dimension accordingly.")
        initial_action = initial_action[:B, :, :]

    trajectory, trajectory_all = policy.predict_action(obs_dict, initial_action)

    return torch.tensor(trajectory, device=device), trajectory_all  # no extend batch dimension.

def sync_if_cuda(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def make_subset_dataloader(dataset, start_idx, num_samples, batch_size, base_loader=None):
    end_idx = min(start_idx + num_samples, len(dataset))

    if end_idx <= start_idx:
        raise ValueError(
            f"Invalid subset range: start_idx={start_idx}, "
            f"num_samples={num_samples}, dataset_len={len(dataset)}"
        )

    indices = list(range(start_idx, end_idx))
    subset = torch.utils.data.Subset(dataset, indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
    }

    if base_loader is not None:
        loader_kwargs.update({
            "num_workers": getattr(base_loader, "num_workers", 0),
            "collate_fn": getattr(base_loader, "collate_fn", None),
            "pin_memory": getattr(base_loader, "pin_memory", False),
        })

    return torch.utils.data.DataLoader(subset, **loader_kwargs)


def run_diffusion_inference_only(policy, dataloader, config_dict, device):
    num_samples = 0

    for data in dataloader:
        map_cond = data["map"].to(device)
        env_cond = data["env"].to(device)

        start = env_cond[:, :2]
        goal = env_cond[:, 2:]

        current_bs = start.shape[0]

        initial_action = torch.randn(
            (
                current_bs,
                config_dict["horizon"],
                config_dict["action_dim"],
            ),
            device=device,
        )

        generate_batched_path(
            policy,
            start,
            goal,
            map_cond,
            initial_action,
            current_bs,
        )

        num_samples += current_bs

    return num_samples


def speed_test_diffusion_policy(
    policy,
    warmup_dataloader,
    eval_dataloader,
    config_dict,
    device,
):
    """
    Warm up on 1000 samples, then measure FPS on the next 100 samples.
    """

    if hasattr(policy, "model"):
        policy.model.eval()

    print("\nStarting speed test...")

    with torch.inference_mode():
        # Warm-up phase: not timed
        print("Running warm-up samples...")
        warmup_samples = run_diffusion_inference_only(
            policy,
            warmup_dataloader,
            config_dict,
            device,
        )

        sync_if_cuda(device)

        # Timed phase
        print("Running timed FPS evaluation...")
        sync_if_cuda(device)
        start_time = time.perf_counter()

        eval_samples = run_diffusion_inference_only(
            policy,
            eval_dataloader,
            config_dict,
            device,
        )

        sync_if_cuda(device)
        elapsed_time = time.perf_counter() - start_time

    fps = eval_samples / elapsed_time

    print("\n=== Speed Test Results ===")
    print(f"Warm-up samples: {warmup_samples}")
    print(f"Timed samples: {eval_samples}")
    print(f"Elapsed time: {elapsed_time:.4f} sec")
    print(f"FPS: {fps:.2f} samples/sec")

    return {
        "warmup_samples": warmup_samples,
        "eval_samples": eval_samples,
        "elapsed_time": elapsed_time,
        "fps": fps,
    }


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
    test_path = args.test_path
    test_bs = 1
    num_warmup = args.num_warmup
    num_eval = args.num_eval

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

       # Speed-test settings
    warmup_samples = num_warmup
    eval_samples = num_eval
    total_required_samples = warmup_samples + eval_samples

    if len(test_dataset) < total_required_samples:
        raise ValueError(
            f"Dataset has only {len(test_dataset)} samples, "
            f"but speed test requires {total_required_samples} samples."
        )

    warmup_dataloader = make_subset_dataloader(
        dataset=test_dataset,
        start_idx=0,
        num_samples=warmup_samples,
        batch_size=test_bs,
        base_loader=test_dataloader,
    )

    eval_dataloader = make_subset_dataloader(
        dataset=test_dataset,
        start_idx=warmup_samples,
        num_samples=eval_samples,
        batch_size=test_bs,
        base_loader=test_dataloader,
    )

    # Run speed test
    results = speed_test_diffusion_policy(
        policy=policy,
        warmup_dataloader=warmup_dataloader,
        eval_dataloader=eval_dataloader,
        config_dict=config_dict,
        device=device,
    )

    return results


if __name__ == "__main__":
    results = main()