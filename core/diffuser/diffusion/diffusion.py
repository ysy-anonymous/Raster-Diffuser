import numpy as np
import torch
import torch.nn.functional as F

from utils.normalizer import LinearNormalizer

# dataset utils
from core.diffuser.datasets.distance_map_gen import distance_to_obstacle, signed_distance_field

# cost utils for multi-hypothesis refiner trajectory planner
from utils.cost_utils import collision_cost

class PlaneDiffusionPolicy:
    def __init__(
        self,
        model,
        noise_scheduler,
        config,
        device,
    ):
        self.device = device
        self.net = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.norm_stats = config["normalizer"]
        self.config = config
        self.use_single_step_inference = False

        # if the network has 'set_noise_scheduler' function, call it
        method = getattr(self.net, 'set_noise_scheduler', None)
        if callable(method):
            method(self.noise_scheduler)
            
        # move network to device (call this after setting noise scheduler if needed)
        self.net.to(self.device).eval() # evaluation mode


    def predict_action(self, obs_dict: dict, initial_action, generator=None):
        """
        obs_dict：
        - "sample": [obs_horizon, obs_dim]
        - "env": [2 * obs_dim]
        - "map": [1, 8, 8]
        """

        obs_seq = obs_dict["sample"]  # shape: [obs_horizon, obs_dim] or [B, obs_horizon, obs_dim]
        nobs = self.normalizer.normalize_data(obs_seq, stats=self.norm_stats)
        nobs = nobs.flatten()
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)
        # load observation

        if len(obs_dict['env'].shape)==1: # for batch=1, squeezed input
            st_point = obs_dict['env'][:2]
            goal_point = obs_dict['env'][2:]
        else: # for batched input
            st_point = obs_dict['env'][:, :2]
            goal_point = obs_dict['env'][:, 2:]
        
        st_point = self.normalizer.normalize_data(st_point, self.norm_stats)  # normalize start/goal point
        goal_point = self.normalizer.normalize_data(goal_point, self.norm_stats)
        st_goal = np.concatenate([st_point, goal_point], axis=-1)

        if len(st_goal.shape) == 1:
            env_cond = torch.from_numpy(st_goal).to(self.device, dtype=torch.float32).unsqueeze(0)
        else:
            env_cond = torch.from_numpy(st_goal).to(self.device, dtype=torch.float32) # no need to add batch dimension
        
        # load map
        if len(obs_dict['map'].shape) == 3:
            obs_mask = torch.from_numpy(obs_dict["map"]).to(self.device, dtype=torch.float32).unsqueeze(0)  # [1, 1, 8, 8]
        else:
            obs_mask = torch.from_numpy(obs_dict['map']).to(self.device, dtype=torch.float32) # [B, 1, 8, 8]

        if self.config['dataset']['include_distance_map']:
            cell_size = self.config['dataset']['cell_size']
            boundary_zero = self.config['dataset']['boundary_zero']
            if self.config['dataset']['sdf_only']:
                sdf_only = signed_distance_field(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
                map_cond = sdf_only # For sdf-only setting, we don't normalize the sdf values.
            elif self.config['dataset']['dmap_only']:
                dmap_only = distance_to_obstacle(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
                map_cond = dmap_only # For dmap-only setting, we don't normalize the distance map values.
            else:
                dst = distance_to_obstacle(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
                sdf = signed_distance_field(obs_mask, cell_size=cell_size, boundary_zero=boundary_zero)
                map_cond = torch.cat([obs_mask, dst, sdf], dim=1) # (B, 3, 8, 8)
                map_cond[:, 1:] = map_cond[:, 1:] / (map_cond[:, 1:].amax(dim=(-2, -1), keepdim=True) + 1e-6)
        else:
            map_cond = obs_mask
                    
        map_cond = F.interpolate(map_cond, scale_factor=self.config['dataset']['input_upsample'], mode='bilinear') # upsample binary obstacle map
        noisy_action = initial_action
        naction = noisy_action

        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        timesteps = (
            self.noise_scheduler.timesteps[:1] if self.use_single_step_inference else self.noise_scheduler.timesteps
        )
        
        if naction.shape[0] == 1: # for the 1 batch input
            action_all = [naction.detach().cpu().numpy()[0]]
        else:
            action_all = [naction.detach().cpu().numpy()] # [B, 32, 2]

        for t in timesteps:
            with torch.no_grad():
                net_out = self.net(
                    sample=naction,
                    timestep=t,
                    map_cond=map_cond,
                    env_cond=env_cond,
                )
                
            if isinstance(net_out, tuple):
                if len(net_out) == 2:
                    noise_pred, x0_k = net_out
                elif len(net_out) == 3:
                    noise_pred, x0_k, aux = net_out
                    # noise_pred, x0_k, logits = net_out
            else:
                noise_pred = net_out
            
            # if you use multi-hypothesis trajectory planner, you need to choose the hypothesis
            if len(noise_pred.shape) == 3: # (B, T, 2)
                pass
            elif len(noise_pred.shape) == 4: # (B, K, T, 2)
                B = noise_pred.shape[0]
                
                # 1. Cost metric based selection
                cost = collision_cost(obs_mask, x0_k) 
                k_star = cost.argmin(dim=1)
                noise_pred = noise_pred[torch.arange(B), k_star]
                
                # 2. Trained neural Network based selection
                # k_pred = logits.argmax(dim=1)  # (B,)
                # noise_pred = noise_pred[torch.arange(B), k_pred]  # (B,T,2)
                
            if self.use_single_step_inference:
                naction = noisy_action - noise_pred
            else:
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=naction,
                    generator=generator, # <-- important
                ).prev_sample
                
            if naction.shape[0] == 1:
                action_all.append(naction.detach().cpu().numpy()[0].copy())
            else:
                action_all.extend(naction.detach().cpu().numpy().copy()) # extend batch size


        if naction.shape[0] == 1:
            action_pred = naction.detach().cpu().numpy()[0]
        else:
            action_pred = naction.detach().cpu().numpy()

        action_pred = self.normalizer.unnormalize_data(action_pred, stats=self.norm_stats)
        
        action_all_unnorm = []
        for action_step in action_all:
            action_step_unnorm = self.normalizer.unnormalize_data(action_step, stats=self.norm_stats)
            action_all_unnorm.append(action_step_unnorm)

        return action_pred, action_all_unnorm  # shape: [pred_horizon, action_dim]


    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_weight = state_dict.get("model_state_dict", state_dict)

        self.net.load_state_dict(model_weight)
