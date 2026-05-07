# Code for testing potential-based diffusion models in proposed dataset task.
import torch
import torch.nn.functional as F
from utils.normalizer import LinearNormalizer
from core.pb_diffusion.diffusion.diffusion_pb import GaussianDiffusionPB
from core.diffuser.datasets.distance_map_gen import distance_to_obstacle, signed_distance_field
from core.pb_diffusion.utils import to_np


class PBDiffusionPolicy:
    def __init__(
        self,
        model: GaussianDiffusionPB,
        config,
        device,
    ):
        self.device = device
        self.diffusion_model = model
        self.normalizer = LinearNormalizer()
        
        self.action_dim = config['diffusion_config']['action_dim'] # set this!
        self.horizon = config['diffusion_config']['horizon']
        self.use_ddim = config['policy_config']['use_ddim']
        
        self.norm_stats = config["normalizer"]
        self.config = config
        self.use_single_step_inference = False
            
        # move network to device
        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()


    # Diffusion Policy should implement the Reverse Diffusion Process for Inference
    def predict_x0(self, map_cond, obs_cond, return_diffusion=False):
        
        assert not self.diffusion_model.training
        self.diffusion_model.horizon = self.horizon
        
        ## Run Reverse Diffusion Process
        if type(self.diffusion_model) == GaussianDiffusionPB:
            sample = self.diffusion_model(map_cond, obs_cond, return_diffusion=return_diffusion, use_ddim=self.use_ddim)
            
            if return_diffusion:
                sample, diffusion = sample
                sample = to_np(sample); diffusion = to_np(diffusion)
                return sample, diffusion
            
            sample = to_np(sample)
            return sample
        else:
            NotImplementedError(f'type: {type(self.diffusion_model)}')
                
    
    def predict_action(self, obs_dict: dict):
        """
        obs_dict 结构：
        - "sample": [obs_horizon, obs_dim]
        - "env": [2 * obs_dim] 的拼接向量
        - "map": [1, 8, 8]
        """


        obs_seq = obs_dict["sample"]  # shape: [obs_horizon, obs_dim]
        nobs = self.normalizer.normalize_data(obs_seq, stats=self.norm_stats)
        nobs = nobs.flatten()
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

        if len(obs_dict['env'].shape) == 1: # If no batched input, add batch dimension
            # load observation
            env_cond = torch.from_numpy(obs_dict["env"]).to(self.device, dtype=torch.float32).unsqueeze(0)  # [1, 2*obs_dim]
        else:
            env_cond = torch.from_numpy(obs_dict["env"]).to(self.device, dtype=torch.float32)  # [B, 2*obs_dim]

        # load map
        if len(obs_dict['map'].shape) == 3:
            obs_mask = torch.from_numpy(obs_dict["map"]).to(self.device, dtype=torch.float32).unsqueeze(0)  # [1, 1, 8, 8]
        else:
            obs_mask = torch.from_numpy(obs_dict['map']).to(self.device, dtype=torch.float32) # [B, 1, 8, 8]

        if self.config['dataset']['include_distance_map']:
            cell_size = self.config['dataset']['cell_size']
            boundary_zero = self.config['dataset']['boundary_zero']
            dst = distance_to_obstacle(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
            sdf = signed_distance_field(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
            map_cond = torch.cat([obs_mask, dst, sdf], dim=1) # (B, 3, 8, 8)
            map_cond[:, 1:] = map_cond[:, 1:] / (map_cond[:, 1:].amax(dim=(-2, -1), keepdim=True) + 1e-6)
        else:
            map_cond = obs_mask
        
        if self.config['dataset']['input_upsample'] != 1.0:
            map_cond = F.interpolate(map_cond, scale_factor=self.config['dataset']['input_upsample'], mode='bilinear') # upsample binary obstacle map
        
        action_pred, action_pred_all = self.predict_x0(map_cond, env_cond, return_diffusion=True)
        action_pred = self.normalizer.unnormalize_data(action_pred, stats=self.norm_stats)
        
        action_all_unnorm = []
        for action_step in action_pred_all:
            action_step_unnorm = self.normalizer.unnormalize_data(action_step, stats=self.norm_stats)
            action_all_unnorm.append(action_step_unnorm)

        return action_pred, action_all_unnorm  # shape: [pred_horizon, action_dim]


    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        step = state_dict['step']
        ema = state_dict['ema']
        model = state_dict['model']
        self.diffusion_model.load_state_dict(ema)