# Raster-Diffuser

Anonymous code release for:

**Raster-Diffuser: Improving Diffusion Motion Planning with Raster-Guided Refinement**

This repository contains the implementation of **Raster-Diffuser**, a diffusion-based motion planning framework for collision-aware trajectory generation under dense 2D obstacle-map conditioning. The key idea is to rasterize the model's intermediate clean trajectory estimate during reverse diffusion and use this spatial representation for iterative map-guided refinement.

This repository is anonymized for double-blind review.

---

## Overview

Raster-Diffuser addresses the representation mismatch between:

- waypoint-based trajectory sequences, and
- dense 2D obstacle maps.

At each diffusion timestep, the model reconstructs an intermediate clean trajectory estimate, rasterizes it into the obstacle-map coordinate system using Gaussian trajectory rasterization, and refines the trajectory using spatial map-conditioned features.

This repository includes:

- Raster-Diffuser implementation
- 2D-map-conditioned diffusion baselines
- RRT*-based synthetic dataset generation
- training scripts
- evaluation scripts
- configuration files for the main experiments
- reproduction instructions for the main results

---

## Repository Structure

```text
Raster-Diffuser/
├── anonymous_check.sh
├── attributes/
├── core/
│   ├── comp_diffusion/
│   ├── diffuser/
│   ├── pb_diffusion/
│   └── rediffuser/
├── data_generator_d/
│   ├── data_dynamic_env.py
│   ├── data_generator_grid.py
│   ├── data_generator.py
│   ├── RRT_star_grid.py
│   └── RRT_star.py
├── dataset/
│   ├── test_data.py
│   ├── test_dataset.py
│   ├── test.py
│   ├── train_data_set_2000.npy
│   ├── train_data_set_flatten.npy
│   └── train_data_set.npy
├── run/
│   ├── train_diffusion.py
│   ├── train_diffusion_ddp.py
│   ├── train_comp_diffusion.py
│   ├── train_comp_ddp.py
│   ├── train_pb_diffusion.py
│   ├── train_pb_ddp.py
│   ├── train_rediffuser.py
│   ├── train_rediffuser_ddp.py
│   ├── test_diffuser_offline.py
│   ├── test_comp_offline.py
│   ├── test_pbdmp_offline.py
│   └── test_rediffuser_offline.py
├── scripts/
│   ├── speed/
│   ├── test/
│   └── train/
├── test/
│   ├── dataset_test.py
│   └── network_embedd_test.py
├── utils/
│   ├── config_utils.py
│   ├── cost_utils.py
│   ├── dataset_utils.py
│   ├── load_utils.py
│   ├── normalizer.py
│   ├── value_utils.py
│   ├── visualizer.py
│   ├── visualizer_2.py
│   └── vis_utils.py
├── environment.yml
├── requirements.txt
├── UPSTREAM_IMPORT.txt
└── README.md
```

Some analysis notebooks are also included for inspection and visualization. They are not required for reproducing the main experiments.

---

## Environment Setup

We recommend using a clean Conda environment.

```bash
conda create -n raster_diffuser python=3.12 -y
conda activate raster_diffuser
pip install -r requirements.txt
```

---

## Hardware

The main experiments were run on NVIDIA RTX 8000 GPUs with 48GB memory per GPU.

Approximate training time for the main `(8 x 8)` experiments:

| Setting | GPUs | Epochs | Approx. training time | Approx. GPU-hours |
|---|---:|---:|---:|---:|
| 2K samples | 1 RTX 8000 | 1000 | 45 minutes | 0.75 |
| 40K samples | 4 RTX 8000 | 1000 | 2-3 hours | 8-12 |
| 100K samples | 4 RTX 8000 | 1000 | 5-6 hours | 20-24 |

Runtime may vary depending on CUDA version, PyTorch version, storage speed, and dataloader settings.

---

## Dataset Path Setup

Before training, set the dataset paths for your local environment.

The dataset paths are configured in:

```text
utils/dataset_utils.py
```

For anonymous review and portability, we recommend using a project-local dataset directory such as:

```text
dataset/
```


A portable version of the dataset selector can be written as:

```python
import os
from pathlib import Path

def select_dataset(dataset_id):
    data_root = Path(os.environ.get("RASTER_DIFFUSER_DATA_ROOT", "dataset"))

    dataset_map = {
        0: "train_data_set.npy",
        1: "train_data_set_2000.npy",
        2: "train_data_set_6257_16x16.npy",
        3: "train_data_set_11210_32x32.npy",
        4: "train_data_set_38345_8x8.npy",
        5: "train_data_set_95792_8x8.npy",
        6: "train_data_set_1053418_8x8.npy",
        7: "train_data_set_97529_16x16_64h.npy",
        8: "train_data_set_95035_32x32_128h.npy",
        9: "train_data_set_8000_8x8.npy",
    }

    if dataset_id not in dataset_map:
        raise ValueError(f"Invalid dataset id: {dataset_id}")

    return str(data_root / dataset_map[dataset_id])
```

The dataset IDs used in the main experiments are:

| Dataset ID | Description |
|---:|---|
| 1 | 8 x 8, approximately 2K training samples |
| 4 | 8 x 8, approximately 40K training samples |
| 5 | 8 x 8, approximately 100K training samples |
| 7 | 16 x 16, approximately 100K training samples, horizon 64 |
| 8 | 32 x 32, approximately 100K training samples, horizon 128 |

---

## Model Configuration

Model configurations are located in:

```text
core/diffuser/config/diffuser_bil.py
core/diffuser/config/DDP/diffuser_bil_ddp.py
```

The default settings correspond to the configurations used in the paper.

For single-GPU training, use:

```text
core/diffuser/config/diffuser_bil.py
```

For distributed multi-GPU training, use:

```text
core/diffuser/config/DDP/diffuser_bil_ddp.py
```

---

## Training Raster-Diffuser

### Single-GPU Training

Example command for training Raster-Diffuser on the 2K setting:

```bash
python3 run/train_diffusion.py \
    --model_id 4 \
    --epochs 1000 \
    --save_step 500 \
    --save_path outputs/checkpoints/raster_diffuser_8x8_2k \
    --config_path core/diffuser/config/diffuser_bil.py \
    --device cuda:0 \
    --dataset_id 1
```

To run the command in the background and save logs:

```bash
nohup python3 run/train_diffusion.py \
    --model_id 4 \
    --epochs 1000 \
    --save_step 500 \
    --save_path outputs/checkpoints/raster_diffuser_8x8_2k \
    --config_path core/diffuser/config/diffuser_bil.py \
    --device cuda:0 \
    --dataset_id 1 \
    > outputs/logs/raster_diffuser_8x8_2k_train.log 2>&1 &
```

### Multi-GPU Training

Example command for distributed training on the 40K setting using four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    run/train_diffusion_ddp.py \
    --model_id 4 \
    --epochs 1000 \
    --save_step 500 \
    --save_path outputs/checkpoints/raster_diffuser_8x8_40k \
    --config_path core/diffuser/config/DDP/diffuser_bil_ddp.py \
    --dataset_id 4 \
    --batch_size 256 \
    --num_workers 2 \
    --pin_memory \
    > outputs/logs/raster_diffuser_8x8_40k_train.log 2>&1
```

Example command for the 100K setting:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    run/train_diffusion_ddp.py \
    --model_id 4 \
    --epochs 1000 \
    --save_step 500 \
    --save_path outputs/checkpoints/raster_diffuser_8x8_100k \
    --config_path core/diffuser/config/DDP/diffuser_bil_ddp.py \
    --dataset_id 5 \
    --batch_size 256 \
    --num_workers 2 \
    --pin_memory \
    > outputs/logs/raster_diffuser_8x8_100k_train.log 2>&1
```

---

## Evaluation

To evaluate a trained Raster-Diffuser model on the offline test set:

```bash
python3 run/test_diffuser_offline.py \
    --model_id 4 \
    --ddp_trained \
    --cp_path outputs/checkpoints/raster_diffuser_8x8_40k/ckpt_final.ckpt \
    --device cuda:0 \
    --num_vis 500 \
    --vis_fname raster_diffuser_8x8_40k_offline \
    --test_path dataset/test_scenarios_20000.npy \
    --test_bs 64
```

To save the evaluation log:

```bash
nohup python3 run/test_diffuser_offline.py \
    --model_id 4 \
    --ddp_trained \
    --cp_path outputs/checkpoints/raster_diffuser_8x8_40k/ckpt_final.ckpt \
    --device cuda:0 \
    --num_vis 500 \
    --vis_fname raster_diffuser_8x8_40k_offline \
    --test_path dataset/test_scenarios_20000.npy \
    --test_bs 64 \
    > outputs/logs/raster_diffuser_8x8_40k_eval.log 2>&1 &
```

---

## Training and Evaluating Baselines

The following baseline implementations are included:

- Diffuser
- Potential-Based Diffusion Motion Planning
- CompDiffuser
- ReDiffuser

Example training scripts are located in:

```text
run/train_comp_diffusion.py
run/train_pb_diffusion.py
run/train_rediffuser.py
run/train_comp_ddp.py
run/train_pb_ddp.py
run/train_rediffuser_ddp.py
```

Example evaluation scripts are located in:

```text
run/test_comp_offline.py
run/test_pbdmp_offline.py
run/test_rediffuser_offline.py
```

Please refer to the corresponding configuration files under `core/` for each baseline.

---

## Data Generation

Synthetic motion-planning datasets can be generated using:

```bash
python3 data_generator_d/data_generator_grid.py
```

The data generation script allows control over:

- map resolution,
- number of training trajectories,
- number of test scenarios,
- obstacle-count range,
- obstacle size range,
- trajectory horizon,
- random seed.

Please edit the configuration section inside `data_generator_d/data_generator_grid.py` before generation, or add command-line arguments if needed for your setup.

---

## Main Metrics

The paper reports the following metrics:

- **Success Rate (SR)**: percentage of generated trajectories that reach the goal without collision.
- **Collision Rate (CR)**: percentage of generated trajectories that collide with obstacles.
- **Failed-to-Reach Rate (FRR)**: percentage of generated trajectories that fail to satisfy start or goal constraints.


---

## Reproducibility Notes

To reproduce the main results:

1. Install the environment.
2. Set `RASTER_DIFFUSER_DATA_ROOT` or update `utils/dataset_utils.py`.
3. Generate or place datasets under the dataset root.
4. Train Raster-Diffuser using the commands above.
5. Evaluate the final checkpoint using the offline evaluation script.

Recommended output directories:

```text
outputs/
├── checkpoints/
├── logs/
└── visualizations/
```

These directories are ignored by version control and can be created locally:

```bash
mkdir -p outputs/checkpoints outputs/logs outputs/visualizations
```

---

## License

Please see the `LICENSE` file for the license of this repository.

If using or redistributing code adapted from upstream repositories, please also check the corresponding upstream license and terms of use.

---