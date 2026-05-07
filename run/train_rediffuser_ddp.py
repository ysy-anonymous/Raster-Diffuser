import os
import shutil
import argparse

import torch
import torch.distributed as dist

from utils.dataset_utils import select_dataset
from utils.load_utils import build_rediff_from_cfg

from core.rediffuser.datasets import plane_dataset_embeed
from core.rediffuser.trainer.rediff_ddp_trainer import ReDiffDDPTrainer


def parse_argument():
    parser = argparse.ArgumentParser(description='training diffusion argument parser')

    parser.add_argument('--model_id', type=int, default=0, help='0: ReDiffuser')
    parser.add_argument('--ddp_trained', action='store_true', help='Enable DistributedDataParallel training')
    parser.add_argument(
        '--config_path',
        type=str,
        default='/exhdd/seungyu/diffusion_motion/core/networks/rediffuser/configs/rediff_ddp_cfg.py',
        help='set config file path'
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=1,
        help='dataset id for training, '
             '0: 8x8 100 dataset, 1: 8x8 2000 dataset, 2: 16x16 6257 dataset, '
             '3: 32x32 11210 dataset, 4: 8x8 38345 dataset, 5: 8x8 95792 dataset, '
             '6: 8x8 1053418 dataset'
    )

    args = parser.parse_args()
    return args


def setup_ddp():
    """
    Initialize DDP from torchrun environment variables.
    Returns:
        use_ddp, rank, world_size, local_rank
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """
    Main function to load model and train the model
    """
    args = parse_argument()
    train_model = args.model_id
    dataset_id = args.dataset_id

    # Enable DDP only if both:
    # 1) user asked for it with --ddp_trained
    # 2) script was launched with torchrun
    launched_with_torchrun = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    use_ddp = args.ddp_trained and launched_with_torchrun

    if use_ddp:
        use_ddp, rank, world_size, local_rank = setup_ddp()
    else:
        rank, world_size, local_rank = 0, 1, 0

    is_main_process = rank == 0

    # build network and configuration
    diffusion_model, config_dict = build_rediff_from_cfg(train_model, ddp=use_ddp)

    # set device
    if use_ddp:
        train_device = f"cuda:{local_rank}"
    else:
        train_device = config_dict['trainer'].get(
            'train_device',
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    config_dict['trainer']['train_device'] = train_device
    diffusion_model = diffusion_model.to(train_device)

    # prepare result directory only on rank 0
    save_path = config_dict['trainer']['results_folder']
    if is_main_process:
        os.makedirs(save_path, exist_ok=True)
        shutil.copyfile(args.config_path, os.path.join(save_path, 'config.py'))

    # wait until rank 0 finishes directory/config setup
    if use_ddp:
        dist.barrier()

    # dataset
    dataset_path = select_dataset(dataset_id)
    dataset_config = config_dict['dataset']
    dataset = plane_dataset_embeed.PlanePlanningDataSets(
        dataset_path=dataset_path,
        **dataset_config
    )

    # make sure trainer gets DDP flag
    trainer_cfg = dict(config_dict['trainer'])
    trainer_cfg['use_ddp'] = use_ddp

    trainer = ReDiffDDPTrainer(
        diffusion_model=diffusion_model,
        dataset=dataset,
        **trainer_cfg
    )

    try:
        trainer.train(n_train_steps=config_dict['trainer']['n_train_steps'])
    finally:
        if hasattr(trainer, "cleanup"):
            trainer.cleanup()
        else:
            cleanup_ddp()


if __name__ == '__main__':
    main()