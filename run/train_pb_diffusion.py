import torch
from utils.dataset_utils import select_dataset
from utils.load_utils import build_pb_diff_from_cfg

from core.pb_diffusion.trainer.pb_trainer import TrainerPBDiff # Trainer for PB Diffusion

from core.pb_diffusion.datasets import plane_dataset_embeed 

import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='training diffusion argument parser')
    
    parser.add_argument('--model_id', type=int, default=0, help='0: Directly Adapted PBDMP, 1: PBDMP + BILatent Fusion (Not Yet Available), 2: V2 PBDMP')
    parser.add_argument('--config_path', type=str, default='/exhdd/seungyu/diffusion_motion/core/networks/pb_diffusion/configs/pb_diffusion_cfg.py', help='set config file path')
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
    
    # build network and configuration
    diffusion_model, config_dict = build_pb_diff_from_cfg(train_model)
    diffusion_model = diffusion_model.to(config_dict['trainer']['train_device'])
    
    save_path = config_dict['trainer']['results_folder']
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    config_path = args.config_path
    shutil.copyfile(config_path, os.path.join(save_path, 'config.py'))


    
    dataset_path = select_dataset(dataset_id)
    dataset_config = config_dict['dataset']
    dataset = plane_dataset_embeed.PlanePlanningDataSets(
        dataset_path=dataset_path, **dataset_config
    )
    
    trainer = TrainerPBDiff(diffusion_model=diffusion_model, dataset=dataset, **config_dict['trainer'])
    trainer.train(n_train_steps=config_dict['trainer']['n_train_steps'])

if __name__ == '__main__':
    main()