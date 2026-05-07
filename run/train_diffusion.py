import torch
from utils.dataset_utils import select_dataset

from core.diffuser.trainer.plane_diffusion_trainer_embeed import PlaneDiffusionTrainer # Import Trainer
from core.diffuser.datasets import plane_dataset_embeed
from utils.load_utils import build_networks_from_config

import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='training diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: default model, 1: transformer decoder, 2: default + transformer, ' \
    '3: diffusion planner, 4: bilatent diffusion planner, 5: multi-hypothesis diffusion planner, 6: trajectory planner plus, 7: multi resolution timestep trajectory planner')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--save_step', type=int, default=100, help='number of steps to save model')
    parser.add_argument('--save_path', type=str, default='/exhdd/seungyu/diffusion_motion/trained_weights/conv', help='trained model weight path')
    parser.add_argument('--config_path', type=str, default='/exhdd/seungyu/diffusion_motion/config/plane_test_embedded.py', help='model_config_path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run model')
    parser.add_argument('--dataset_id', type=int, default=1, help='dataset id for training, ' \
    '0: 8x8 100 dataset, 1: 8x8 2000 dataset, 2: 16x16 6257 dataset, 3: 32x32 11210 dataset, 4: 8x8 38345 dataset, 5: 8x8 95792 dataset,\
    6: 8x8 1053418 dataset')
    args = parser.parse_args()
    
    return args
    
def main():
    """
    Main function to load model and train the model
    """
    import os
    import shutil

    # parse argument
    args = parse_argument()
    train_model = args.model_id
    dataset_id = args.dataset_id
    device = args.device

    # build config and networks
    config, config_dict, net = build_networks_from_config(train_model)
    dataset_path = select_dataset(dataset_id)
    
    dataset_config = config_dict['dataset']
    dataset = plane_dataset_embeed.PlanePlanningDataSets(
        dataset_path=dataset_path, **dataset_config
    )

    # build trainer
    trainer = PlaneDiffusionTrainer(
        config=config,
        net=net,
        dataset=dataset,
        device=device
    )

    config_path = args.config_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shutil.copyfile(config_path, os.path.join(save_path, 'config.py'))

    trainer.train(num_epochs=args.epochs, save_ckpt_epoch=args.save_step, save_path=save_path)
    trainer.save_checkpoint(f"{save_path}/ckpt_final.ckpt")

if __name__ == '__main__':
    main()