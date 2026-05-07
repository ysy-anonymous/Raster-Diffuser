import os
import shutil
import argparse
import torch
import torch.distributed as dist

from utils.dataset_utils import select_dataset
from utils.load_utils import build_comp_diff_from_cfg

from core.comp_diffusion.cd_stgl_sml_dfu.stgl_sml_training_ddp import Stgl_Sml_Trainer_DDP
from core.comp_diffusion.datasets.rrt_map import RRTCompDataset


def parse_argument():
    parser = argparse.ArgumentParser(description='training diffusion argument parser')

    parser.add_argument(
        '--model_id',
        type=int,
        default=0,
        help='0: Directly Adapted Comp Diffuser, 1: Comp Diffuser V2'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='/exhdd/seungyu/diffusion_motion/core/networks/comp_diffusion/configs/comp_diffusion_ddp_cfg.py',
        help='set config file path'
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=1,
        help='dataset id for training, '
             '0: 8x8 100 dataset, 1: 8x8 2000 dataset, 2: 16x16 6257 dataset, '
             '3: 32x32 11210 dataset, 4: 8x8 38345 dataset, '
             '5: 8x8 95792 dataset, 6: 8x8 1053418 dataset'
    )
    parser.add_argument(
        '--use_ddp',
        action='store_true',
        help='Enable DistributedDataParallel training'
    )

    return parser.parse_args()


def is_main_process(rank: int) -> bool:
    return rank == 0


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_argument()

    ddp_flag=False
    if args.use_ddp:
        ddp_flag = True
        rank, world_size, local_rank = setup_ddp()
        device = f"cuda:{local_rank}"
        print("DDP Enabled: Use DDP flag")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        train_model = args.model_id
        dataset_id = args.dataset_id

        # build network and configuration
        diffusion_model, config_dict = build_comp_diff_from_cfg(train_model, ddp=ddp_flag)

        # override per-process device
        config_dict["trainer"]["device"] = device

        save_path = config_dict["trainer"]["results_folder"]

        if is_main_process(rank):
            os.makedirs(save_path, exist_ok=True)
            shutil.copyfile(args.config_path, os.path.join(save_path, "config.py"))

        if args.use_ddp:
            dist.barrier()

        dataset_path = select_dataset(dataset_id)
        dataset_config = config_dict["dataset"]
        dataset = RRTCompDataset(
            dataset_path=dataset_path,
            **dataset_config
        )

        trainer = Stgl_Sml_Trainer_DDP(
            diffusion_model=diffusion_model,
            dataset=dataset,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            use_ddp=args.use_ddp,
            **config_dict["trainer"],
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )

        trainer.train(n_train_steps=config_dict["trainer"]["n_train_steps"])

    finally:
        if args.use_ddp:
            cleanup_ddp()


if __name__ == '__main__':
    main()