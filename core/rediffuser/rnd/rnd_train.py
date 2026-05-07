import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import time

from rnd import RNDModel


parser = argparse.ArgumentParser()
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
robuster = "rnd"

# change keyword and horizon here!
# keyword = "rrt_8x8_100k"
keyword = 'rrt_8x8_40k'
# keyword = 'rrt_8x8_2k'
H = 32
output_dim = 32

# rnd
input_size = (2, H)
dataset_path = '/exhdd/seungyu/diffusion_motion/core/rediffuser/datasets/RNDdataset'
save_base_dir = "/exhdd/seungyu/diffusion_motion/core/rediffuser/rnd"

batch_size = 512
n_epochs = 200
learning_rate = 1e-4 # for 2k samples, we train with higher learning rate (4e-4) for faster convergence, since the dataset is smaller and less noisy. For 100k samples, we use 1e-4 learning rate.
horizon_gap = 32 # When set to 32, there is no augmentation (see Hindsight Truncated Plan)
seed = 42
use_hindsight_truncated_plan = False # As all the horizon is fixed to length 32, no hindsight truncated plan augmentation is needed.

save_dir = os.path.join(save_base_dir, "RNDModel", "model_{}_{}_{}e".format(robuster, keyword, n_epochs))

if seed:
    torch.manual_seed(seed)
    np.random.seed(seed)

def train_rnd():    
    model = RNDModel(input_size=input_size, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.predictor.parameters(), lr=learning_rate)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
        logger.write("epoch\tloss\n")
    
    orig_dataset = np.load(dataset_path + "/" + keyword + "_H{}".format(H) + ".npy")
    orig_datasize = orig_dataset.shape
    orig_dataset = np.transpose(orig_dataset.reshape(orig_datasize[0], int(orig_datasize[1]/2), 2), (0, 2, 1))
    orig_datasize = orig_dataset.shape
    print("original dataset size is {}".format(orig_datasize))
    

    if use_hindsight_truncated_plan:
        # Hindsight truncated plan
        print("Hindsight truncated plan:\nConfiguring...")
        dataset = orig_dataset.copy()
        
        for _ in range(int(H/horizon_gap)):
            
            temp_dataset = orig_dataset.copy()
            res_horizons = np.random.randint(1, int(H/horizon_gap)+1, size=(orig_datasize[0],)) * horizon_gap
            
            for (i, res_horizon) in enumerate(res_horizons):
                if res_horizon != H:
                    temp_dataset[i, :, :] = np.concatenate(
                        (temp_dataset[i, :, :1], 
                        temp_dataset[i, :, -(res_horizon-1):], 
                        np.tile(temp_dataset[i, :, :1], (1, int(H-res_horizon)))),
                        axis=1
                    )
            dataset = np.concatenate((dataset, temp_dataset))

            # res_horizons = torch.randint(1, int(H/horizon_gap)+1, size=(datasize[0],)) * horizon_gap
            # for (i, res_horizon) in enumerate(res_horizons):
            #     if res_horizon != H:
            #         dataset[i, :, :] = torch.concatenate(
            #             (dataset[i, :, :1], dataset[i, :, -(res_horizon-1):], torch.zeros((4, int(H-res_horizon)))), 
            #             dim=1
            #         )
        
        np.random.shuffle(dataset)
        data_shape = dataset.shape
        print("Dataset has been expanded by hindsight truncated plan with the shape of {}".format(data_shape))
    else:
        dataset = orig_dataset
        data_shape = dataset.shape
    
    dataloader = DataLoader(TensorDataset(torch.FloatTensor(dataset)), batch_size=batch_size, num_workers=4, 
                            shuffle=True, pin_memory=True)
    
    idxes = np.random.randint(0, data_shape[0], size=10000)
    testset = torch.FloatTensor(dataset[idxes]).to(device)

    with torch.no_grad():
        test_loss = model(testset).mean()
        
        print("epoch: 0\tloss: {}".format(test_loss))
        with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
            logger.write("0\t{}\n".format(test_loss))
    
    for epoch in range(n_epochs):
        for batch in tqdm(dataloader):
            
            batch = batch[0].to(device)
            loss = model(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            test_loss = model(testset).mean()
            
            print("epoch: {}\tloss: {}".format(epoch+1, test_loss))
            with open(os.path.join(save_dir, "epoch_loss.txt"), "a") as logger:
                logger.write("{}\t{}\n".format(epoch+1, test_loss))
    
    torch.save(model.predictor.state_dict(), os.path.join(save_dir, "predictor.pth"))
    torch.save(model.target.state_dict(), os.path.join(save_dir, "target.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))


if __name__ == "__main__":
    train_rnd()