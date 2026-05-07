import os
import shutil
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils.dataset_utils import select_dataset
from core.diffuser.trainer.ddp_trainer import PlaneDiffusionTrainer
from core.diffuser.datasets import plane_dataset_embeed
from utils.load_utils import build_networks_from_config


def parse_argument():
    parser = argparse.ArgumentParser(description="DDP training for diffusion model")

    parser.add_argument(
        '--model_id',
        type=int,
        default=0,
        help='0: default model, 1: transformer decoder, 2: default + transformer, '
             '3: diffusion planner, 4: bilatent diffusion planner, '
             '5: multi-hypothesis diffusion planner, 6: trajectory planner plus, '
             '7: multi resolution timestep trajectory planner'
    )
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/conv')
    parser.add_argument('--config_path', type=str, default='/exhdd/seungyu/diffusion_motion/config/plane_test_embedded.py')
    parser.add_argument('--dataset_id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--master_port', type=str, default='29500')

    return parser.parse_args()


def setup_ddp():
    """
    Initialize DDP from torchrun environment variables.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return local_rank, rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def main():
    args = parse_argument()

    # recommended for many concurrent processes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # build config and network
    train_model = args.model_id
    dataset_id = args.dataset_id
    ddp=False
    if world_size > 1:
        ddp=True
    config, config_dict, net = build_networks_from_config(train_model, ddp=ddp)
    net = net.to(device)

    # wrap model with DDP
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # dataset
    dataset_path = select_dataset(dataset_id)
    dataset_config = config_dict['dataset']

    dataset = plane_dataset_embeed.PlanePlanningDataSets(
        dataset_path=dataset_path,
        **dataset_config
    )

    # distributed sampler
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0)
    )

    save_path = args.save_path
    config_path = args.config_path

    if is_main_process(rank):
        os.makedirs(save_path, exist_ok=True)
        shutil.copyfile(config_path, os.path.join(save_path, 'config.py'))

    dist.barrier()

    # build trainer
    trainer = PlaneDiffusionTrainer(
        config=config,
        net=net,
        dataset=dataset,            # keep if your trainer still expects it
        dataloader=train_loader,    # add this to trainer if possible
        sampler=train_sampler,      # add this to trainer if possible
        device=None,
        rank=rank,
        world_size=world_size
    )

    trainer.train(
        num_epochs=args.epochs,
        save_ckpt_epoch=args.save_step,
        save_path=save_path
    )

    if is_main_process(rank):
        # IMPORTANT: save underlying model, not wrapper
        trainer.save_checkpoint(f"{save_path}/ckpt_final.ckpt")

    cleanup_ddp()


if __name__ == '__main__':
    main()