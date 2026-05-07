import numpy as np
import os
import torch
from core.rediffuser.networks.diffuser.guides.policies import Policy
from utils.load_utils import build_rediff_from_cfg
from core.rediffuser.datasets import plane_dataset_embeed

device = torch.device("cuda:2")
print("device for generating: ", device)

RRT_MAP = np.array([[0.0, 0.0], [8.0, 8.0]])

## Hyperparameters
# keyword = "rrt_8x8_100k"
# keyword = 'rrt_8x8_40k'
keyword = 'rrt_8x8_2k'
save_path='/exhdd/seungyu/diffusion_motion/core/rediffuser/datasets' # save path for saving rnd data
state_range = RRT_MAP
# data_path = 'dataset/train_data_set_95792_8x8.npy'
# data_path = 'dataset/train_data_set_38345_8x8.npy'
data_path = 'dataset/train_data_set_2000.npy'

# norm_stat = {
#             'min': torch.tensor([5.8019e-05, 7.5751e-05]),
#             'max': torch.tensor([8.0000, 8.0000])
# }

# norm_stat =  {
#             'min': torch.tensor([3.4133e-05, 4.6726e-05]),
#             'max': torch.tensor([8.0000, 8.0000])
#         }

norm_stat = {
            'min': torch.tensor([0.0035, 0.0007]),
            'max': torch.tensor([7.9995, 8.0000])
        }
batch_size = 1024
horizon = 32
model_id = 0

seed = 0
print("Random seed: {}".format(seed))
torch.manual_seed(seed)
np.random.seed(seed)

   
def prepare_inputs(batch):
    action = batch['sample'].to(device, dtype=torch.float32)
    map_cond= batch['map'].to(device, dtype=torch.float32)
    env_cond = batch['env'].to(device, dtype=torch.float32)
                
    return action, map_cond, env_cond

def generate_dataset():
    
    diffusion, config = build_rediff_from_cfg(model_id)
    diffusion = diffusion.to(device)
    policy = Policy(diffusion, config=config, horizon=horizon)
    # Load weights
    # policy.load_weights(ckpt_path='/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt', device=device)
    # policy.load_weights(ckpt_path='/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_40k/run1/state_40000.pt', device=device)
    policy.load_weights(ckpt_path='/exhdd/seungyu/diffusion_motion/trained_weights/rediffuser/run1/state_20000.pt', device=device)
    
    dconfig = config['dataset']
    dataset = plane_dataset_embeed.PlanePlanningDataSets(
        dataset_path=data_path, **dconfig)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, pin_memory=True)
    
    print("Number of iterations: {} for batch size {}".format(len(dataloader), batch_size))    
    RNDdataset = np.empty(shape=(0, 2 * horizon))
 
    # Running for just One Epoch. (Iterate through the whole dataset just once)
    for i, batch in enumerate(dataloader, 0):
        
        # Here we don't use rrt-generated trajectory data.
        _, map_cond, env_cond = prepare_inputs(batch)
        b = map_cond.shape[0] # online batch
        
        traj = policy(map_cond, env_cond, device, norm_stat, batch_repeat=1, dataset_mode=True)
        
        states = traj.cpu().numpy()    # array: (batch_size, H, 2)
        # keep the target state before the initial state
        states = np.concatenate((states[:, -1:, :], states[:, :-1, :]), axis=1).reshape(b, -1)
        print ("states shape: ", states.shape)

        try:
            RNDdataset = np.concatenate((RNDdataset, states))
        except:
            print("Non-compatiable shape for concatenation. RNDdataset shape: {}, states shape: {}".format(RNDdataset.shape, states.shape))
        
        if (i+1) % 16 == 0:
            print("{}-th batch at batch size {} Generated!".format(i, batch_size))

        
    
    np.random.shuffle(RNDdataset)
    try:
        if not os.path.exists(save_path + '/RNDdataset'):
            os.makedirs(save_path + "/RNDdataset")
        np.save(save_path + "/RNDdataset/" + keyword + "_H{}".format(horizon) + ".npy", RNDdataset)
        print("RNDdataset has been shuffled and saved with shape of {}".format(RNDdataset.shape))
    except:
        np.save(save_path + "/" + keyword + "_H{}".format(horizon) + ".npy", RNDdataset)
        print("RNDdataset has been shuffled and saved with shape of {}".format(RNDdataset.shape))

if __name__ == "__main__":
    generate_dataset()