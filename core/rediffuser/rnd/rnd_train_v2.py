import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse

from rnd_v2 import ConditionalRNDModel


parser = argparse.ArgumentParser()
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
robuster = "rnd"

# change keyword and horizon here!
# keyword = "rrt_8x8_100k"
# keyword = "rrt_8x8_40k"
# keyword = "rrt_8x8_2k"
keyword = "rrt_16x16_100k"

# H = 32
H = 64

dataset_path = "/exhdd/seungyu/diffusion_motion/core/rediffuser/datasets/RNDdatasetV2"
save_base_dir = "/exhdd/seungyu/diffusion_motion/core/rediffuser/rnd"

batch_size = 512
n_epochs = 100
learning_rate = 1e-4
horizon_gap = 32
seed = 42
use_hindsight_truncated_plan = False   # fixed horizon = 32, so usually False

save_dir = os.path.join(
    save_base_dir,
    "RNDModelV2",
    "model_{}_{}_{}e".format(robuster, keyword, n_epochs)
)

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def maybe_transpose_traj(traj_seq):
    """
    Expect traj_seq as either:
      (N, H, 2)  -> convert to (N, 2, H)
      (N, 2, H)  -> keep
    """
    if traj_seq.ndim != 3:
        raise ValueError(f"traj_seq must be 3D, got shape {traj_seq.shape}")

    if traj_seq.shape[1] == H and traj_seq.shape[2] == 2:
        traj_seq = np.transpose(traj_seq, (0, 2, 1))
    elif traj_seq.shape[1] == 2 and traj_seq.shape[2] == H:
        pass
    else:
        raise ValueError(
            f"Unexpected traj_seq shape {traj_seq.shape}. "
            f"Expected (N, H, 2) or (N, 2, H) with H={H}."
        )
    return traj_seq


def maybe_fix_mask_shape(map_cond):
    """
    Expected map_cond shape for ConditionalRNDModel:
      (N, C, Hm, Wm)

    If loaded as (N, Hm, Wm), convert to (N, 1, Hm, Wm).
    """
    if map_cond.ndim == 3:
        map_cond = np.expand_dims(map_cond, axis=1)
    elif map_cond.ndim == 4:
        pass
    else:
        raise ValueError(
            f"Unexpected map_cond shape {map_cond.shape}. "
            "Expected (N, H, W) or (N, C, H, W)."
        )
    return map_cond


def hindsight_truncate_only_traj(traj_dataset):
    """
    Applies hindsight truncation only to trajectory sequence.
    This should usually be disabled for your fixed horizon=32 setting.
    Input shape: (N, 2, H)
    """
    print("Hindsight truncated plan:\nConfiguring...")
    orig_dataset = traj_dataset
    orig_shape = orig_dataset.shape

    dataset = orig_dataset.copy()

    for _ in range(int(H / horizon_gap)):
        temp_dataset = orig_dataset.copy()
        res_horizons = np.random.randint(
            1, int(H / horizon_gap) + 1, size=(orig_shape[0],)
        ) * horizon_gap

        for i, res_horizon in enumerate(res_horizons):
            if res_horizon != H:
                temp_dataset[i, :, :] = np.concatenate(
                    (
                        temp_dataset[i, :, :1],
                        temp_dataset[i, :, -(res_horizon - 1):],
                        np.tile(temp_dataset[i, :, :1], (1, int(H - res_horizon))),
                    ),
                    axis=1,
                )

        dataset = np.concatenate((dataset, temp_dataset), axis=0)

    np.random.shuffle(dataset)
    print("Trajectory dataset expanded by hindsight truncation:", dataset.shape)
    return dataset


def train_rnd():
    model = ConditionalRNDModel(
        traj_channels=2,
        rnd_output_dim=256,
        context_dim=4,      # change this if env_cond dim is not 4
        mask_in_channels=1, # change this if map_cond has different channel count => if the diffuser trained with binary mask only, use 1 channels.
        traj_feat_dim=128,
        mask_feat_dim=128,
        ctx_feat_dim=64
    ).to(device)

    # Safer than optimizer = Adam(model.predictor.parameters(), ...)
    # because your ConditionalRNDModel likely has predictor split into several modules.
    predictor_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(predictor_params, lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "epoch_loss.txt"), "w") as logger:
        logger.write("epoch\tloss\n")

    dataset_file = os.path.join(dataset_path, f"{keyword}_H{H}.npz")
    data = np.load(dataset_file)

    traj_seq = data["traj_seq"].astype(np.float32)
    map_cond = data["map_cond"].astype(np.float32)
    env_cond = data["env_cond"].astype(np.float32)

    traj_seq = maybe_transpose_traj(traj_seq)   # -> (N, 2, H)
    map_cond = maybe_fix_mask_shape(map_cond)   # -> (N, C, Hm, Wm)

    print("Loaded dataset:")
    print("  traj_seq :", traj_seq.shape)
    print("  map_cond :", map_cond.shape)
    print("  env_cond :", env_cond.shape)

    # Optional safety check for env_cond
    if env_cond.ndim != 2:
        raise ValueError(
            f"env_cond must be 2D, got shape {env_cond.shape}. "
            "Expected (N, context_dim)."
        )

    if use_hindsight_truncated_plan:
        # Only trajectory is changed. For fixed horizon 32, usually keep this False.
        traj_aug = hindsight_truncate_only_traj(traj_seq)

        # Repeat condition tensors to match augmented trajectory count
        repeat_factor = traj_aug.shape[0] // traj_seq.shape[0]
        map_aug = np.tile(map_cond, (repeat_factor, 1, 1, 1))
        env_aug = np.tile(env_cond, (repeat_factor, 1))

        traj_seq = traj_aug
        map_cond = map_aug
        env_cond = env_aug

        print("After augmentation:")
        print("  traj_seq :", traj_seq.shape)
        print("  map_cond :", map_cond.shape)
        print("  env_cond :", env_cond.shape)

    data_size = traj_seq.shape[0]

    dataset = TensorDataset(
        torch.from_numpy(traj_seq),
        torch.from_numpy(map_cond),
        torch.from_numpy(env_cond),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    # Build held-out eval subset from same dataset
    test_size = min(10000, data_size)
    idxes = np.random.choice(data_size, size=test_size, replace=False)

    test_traj = torch.from_numpy(traj_seq[idxes]).to(device)
    test_map = torch.from_numpy(map_cond[idxes]).to(device)
    test_env = torch.from_numpy(env_cond[idxes]).to(device)

    with torch.no_grad():
        test_loss = model(test_traj, test_map, test_env).mean()
        print("epoch: 0\tloss: {}".format(test_loss.item()))
        with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
            logger.write("0\t{}\n".format(test_loss.item()))

    for epoch in range(n_epochs):
        model.train()

        epoch_loss_sum = 0.0
        epoch_count = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            traj_batch, map_batch, env_batch = batch

            traj_batch = traj_batch.to(device, non_blocking=True)
            map_batch = map_batch.to(device, non_blocking=True)
            env_batch = env_batch.to(device, non_blocking=True)

            loss = model(traj_batch, map_batch, env_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_count += 1

        model.eval()
        with torch.no_grad():
            test_loss = model(test_traj, test_map, test_env).mean()

        train_loss_avg = epoch_loss_sum / max(epoch_count, 1)

        print(
            "epoch: {}\ttrain_loss: {:.6f}\ttest_loss: {:.6f}".format(
                epoch + 1, train_loss_avg, test_loss.item()
            )
        )

        with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
            logger.write("{}\t{}\n".format(epoch + 1, test_loss.item()))

    # Save whole model and optimizer
    torch.save(model.state_dict(), os.path.join(save_dir, "conditional_rnd_model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))

    # Optional: save predictor-only trainable weights
    predictor_state = {
        k: v for k, v in model.state_dict().items()
        if "pred_" in k or "predictor" in k
    }
    torch.save(predictor_state, os.path.join(save_dir, "predictor_only.pth"))

    target_state = {
        k: v for k, v in model.state_dict().items()
        if "target_" in k or "target" in k
    }
    torch.save(target_state, os.path.join(save_dir, "target_only.pth"))

if __name__ == "__main__":
    train_rnd()