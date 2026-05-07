import numpy as np
import os
import torch
from core.rediffuser.networks.diffuser.guides.policies import Policy
from utils.load_utils import build_rediff_from_cfg
from core.rediffuser.datasets import plane_dataset_embeed

device = torch.device("cuda:0")
print("device for generating: ", device)

# RRT_MAP = np.array([[0.0, 0.0], [8.0, 8.0]])
RRT_MAP = np.array([[0.0, 0.0], [16.0, 16.0]])

## Hyperparameters
# keyword = "rrt_8x8_100k"
# keyword = "rrt_8x8_40k"
# keyword = "rrt_8x8_2k"
keyword = "rrt_16x16_100k"

save_path = "/exhdd/seungyu/diffusion_motion/core/rediffuser/datasets"
state_range = RRT_MAP

# data_path = "dataset/train_data_set_95792_8x8.npy"
# data_path = "dataset/train_data_set_38345_8x8.npy"
# data_path = "dataset/train_data_set_2000.npy"
data_path = "dataset/train_data_set_97529_16x16_64h.npy"

# norm_stat = {
#     "min": torch.tensor([3.4133e-05, 4.6726e-05]),
#     "max": torch.tensor([8.0000, 8.0000])
# }

# norm_stat = {
#             'min': torch.tensor([5.8019e-05, 7.5751e-05]),
#             'max': torch.tensor([8.0000, 8.0000])
# }

# norm_stat = {
#     "min": torch.tensor([0.0035, 0.0007]),
#     "max": torch.tensor([7.9995, 8.0000])
# }

norm_stat = {
    "min": [0.0002, 0.0002],
    "max": [16.000, 15.9999]
}

batch_size = 1024
horizon = 64 # 32 -> 64
model_id = 0
DDP_Trained=True

seed = 0
print("Random seed: {}".format(seed))
torch.manual_seed(seed)
np.random.seed(seed)

def prepare_inputs(batch):
    action = batch["sample"].to(device, dtype=torch.float32)
    map_cond = batch["map"].to(device, dtype=torch.float32)
    env_cond = batch["env"].to(device, dtype=torch.float32)
    return action, map_cond, env_cond


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def generate_dataset():
    diffusion, config = build_rediff_from_cfg(model_id, DDP_Trained)
    diffusion = diffusion.to(device)
    diffusion.eval()

    policy = Policy(diffusion, config=config, horizon=horizon)

    # Load weights
    # policy.load_weights(
    #     ckpt_path="/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt",
    #     device=device
    # )
    # policy.load_weights(
    #     ckpt_path="/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_40k/run1/state_40000.pt",
    #     device=device
    # )
    # policy.load_weights(
    #     ckpt_path="/exhdd/seungyu/diffusion_motion/trained_weights/rediffuser/run1/state_20000.pt",
    #     device=device
    # )
    policy.load_weights(
        ckpt_path="/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_16x16_64h_100k/run1/state_100000.pt"
    )

    dconfig = config["dataset"]
    dataset = plane_dataset_embeed.PlanePlanningDataSets(dataset_path=data_path, **dconfig)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )

    print("Number of iterations: {} for batch size {}".format(len(dataloader), batch_size))

    traj_seq_list = []
    traj_flat_list = []
    map_cond_list = []
    env_cond_list = []

    # Running for just one epoch
    for i, batch in enumerate(dataloader, 0):
        _, map_cond, env_cond = prepare_inputs(batch)
        b = map_cond.shape[0]

        with torch.no_grad():
            traj = policy(
                map_cond,
                env_cond,
                device,
                norm_stat,
                batch_repeat=1,
                dataset_mode=True
            )

        # traj: (B, H, 2)
        traj_np = to_numpy(traj).astype(np.float32)

        # Original ReDiffuser-style flattening:
        # move final state to front, then flatten to (B, 2H)
        traj_flat = np.concatenate(
            (traj_np[:, -1:, :], traj_np[:, :-1, :]),
            axis=1
        ).reshape(b, -1).astype(np.float32)

        map_np = to_numpy(map_cond).astype(np.float32)
        env_np = to_numpy(env_cond).astype(np.float32)

        traj_seq_list.append(traj_np)
        traj_flat_list.append(traj_flat)
        map_cond_list.append(map_np)
        env_cond_list.append(env_np)

        print(
            "[{}/{}] traj_seq: {}, traj_flat: {}, map: {}, env: {}".format(
                i + 1,
                len(dataloader),
                traj_np.shape,
                traj_flat.shape,
                map_np.shape,
                env_np.shape
            )
        )

    traj_seq_all = np.concatenate(traj_seq_list, axis=0)
    traj_flat_all = np.concatenate(traj_flat_list, axis=0)
    map_cond_all = np.concatenate(map_cond_list, axis=0)
    env_cond_all = np.concatenate(env_cond_list, axis=0)

    n = traj_seq_all.shape[0]
    perm = np.random.permutation(n)

    traj_seq_all = traj_seq_all[perm]
    traj_flat_all = traj_flat_all[perm]
    map_cond_all = map_cond_all[perm]
    env_cond_all = env_cond_all[perm]

    save_dir = os.path.join(save_path, "RNDdatasetV2")
    os.makedirs(save_dir, exist_ok=True)

    save_file = os.path.join(save_dir, "{}_H{}.npz".format(keyword, horizon))

    np.savez_compressed(
        save_file,
        traj_seq=traj_seq_all,     # (N, H, 2)
        traj_flat=traj_flat_all,   # (N, 2H)
        map_cond=map_cond_all,     # e.g. (N, C, Hm, Wm)
        env_cond=env_cond_all      # e.g. (N, D)
    )

    print("Saved RND dataset to:", save_file)
    print("traj_seq shape :", traj_seq_all.shape)
    print("traj_flat shape:", traj_flat_all.shape)
    print("map_cond shape :", map_cond_all.shape)
    print("env_cond shape :", env_cond_all.shape)


if __name__ == "__main__":
    generate_dataset()