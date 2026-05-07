import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Pdfpages for Visualization
from matplotlib.animation import FuncAnimation, PillowWriter
import os

from core.diffuser.datasets.plane_dataset_embeed import PlanePlanningDataSets
from utils.load_utils import build_noise_scheduler_from_config
from utils.dataset_utils import validate_path_collision_free
from utils.vis_hooks import GradCAM1D, FiLMParamExtractor, CustomAttentionExtractor

class Visualizer():
    def __init__(self, diffusion_model, 
                config_dict, 
                test_datapath,
                extractor_config,
                batch_size,
                device):
        self.diffusion_model = diffusion_model
        self.config_dict = config_dict # config dictionary for initializing networks
        self.stats = self.move_np_dict(config_dict['normalizer']) # norm stats (itself is dictionary that contains python array for stats['min'], stats['max'])
        self.dataset = PlanePlanningDataSets(dataset_path=test_datapath, **config_dict['dataset'])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False) # No Data Shuffling ... 
        self.gt_norm = self.dataset.norm_stats

        self.attn_extractor, self.film_extractor, self.cam_extractor = self._init_extractor(extractor_config['attn_extractor'], extractor_config['film_extractor'], 
                                                                                            extractor_config['cam_extractor'])
        self.cam_target_fn = extractor_config['cam_extractor']['cam_target_fn']

        self.noise_scheduler = build_noise_scheduler_from_config(config_dict) # build noise scheduler
        if hasattr(self.diffusion_model, 'set_noise_scheduler'):
            self.diffusion_model.set_noise_scheduler(self.noise_scheduler)
        
        self.device = device
        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()
    
    def _init_extractor(self, attn_config, film_config, cam_config):
        # 1. attention
        attn_extractor = CustomAttentionExtractor(self.diffusion_model, attn_config['q_name'], attn_config['k_name'], attn_config['num_heads'],
                                                  attn_config['q_is_projection'])
        attn_extractor.register()

        # 2. film
        film_extractor = FiLMParamExtractor(self.diffusion_model, film_config['film_names'])
        film_extractor.register()

        # 3. grad-cam
        cam_extractor = GradCAM1D(self.diffusion_model, cam_config['cam_layer_name'])
        cam_extractor.register()

        return attn_extractor, film_extractor, cam_extractor
    
    def _default_record_config(self):
        return {
            "action_target": True,
            "action_all_target": False,
            "attn_map": False,
            "film_stats": False, # here film_stats includes raw film scale, shift value
            "grad_cam": False,
            "gt_target": False,
            "grid_target": False,
            "st_target": False,
            "goal_target": False,
            "do_gradcam": False,   # controls whether backward() is run
        }

    def _merge_record_config(self, record_config=None):
        cfg = self._default_record_config()
        if record_config is not None:
            cfg.update(record_config)
        return cfg

    def _init_output_buffers(self, record_config):
        out = {}
        for key in [
            "action_target",
            "action_all_target",
            "attn_map",
            "film_stats",
            "grad_cam",
            "gt_target",
            "grid_target",
            "st_target",
            "goal_target",
        ]:
            if record_config.get(key, False):
                out[key] = []
        return out

    def _finalize_output_buffers(self, buffers):
        out = {}
        for k, v in buffers.items():
            try:
                out[k] = np.array(v)
            except Exception:
                # keep Python list for nested dict/object cases like film raw value (torch.Tensor)
                out[k] = v
        return out

    def move_np_dict(self, dict_):
        for k, v in dict_.items():
            dict_[k] = np.array(v)
        return dict_
    
    def move_tensor_dict(self, dict_, device=None):
        if device==None:
            device = 'cpu'
        for k, v in dict_.items():
            dict_[k] = torch.tensor(v, device=device)

        return dict_

    def unnormalize_data(self, ndata):
        ndata = (ndata+1)/2
        data = ndata * (np.array(self.stats['max']) - np.array(self.stats['min'])) + np.array(self.stats['min'])
        return data
    
    def unnormalize_w_gt(self, ndata):
        ndata = (ndata+1)/2
        data = ndata * (np.array(self.gt_norm['max']) - np.array(self.gt_norm['min'])) + np.array(self.gt_norm['min'])
        return data

    def normalize_data(self, ndata):
        n, T, d = ndata.shape  
        flat = ndata.reshape(-1, d) 

        eps = 1e-8
        norm = (flat - self.stats['min']) / (self.stats['max'] - self.stats['min'] + eps)  # → [0, 1]
        norm = norm * 2 - 1  # → [-1, 1]

        norm = norm.reshape(n, T, d)
        return norm

    def run_extractors(self, result, net_out, record_config):
        if record_config.get("attn_map", False):
            result["attention"] = self.attn_extractor.get_attention()

        if record_config.get("film_stats", False):
            result["film_stats"] = self.film_extractor.summarize()

        if record_config.get("do_gradcam", False) and record_config.get("grad_cam", False):
            target = self.cam_target_fn(net_out)
            self.diffusion_model.zero_grad(set_to_none=True)
            target.backward()
            result["gradcam_1d"] = self.cam_extractor.compute()

        return result
    
    def close(self):
        self.attn_extractor.remove()
        self.film_extractor.remove()
        self.cam_extractor.remove()

    def reset(self):
        self.attn_extractor.reset()
        self.film_extractor.reset()
        self.cam_extractor.reset()
    
    def collect_vis(self, data, record_config=None):
        record_config = self._merge_record_config(record_config)

        gt_label = data["sample"]
        map_cond = data["map"]
        env = data["env"]
        B = data['sample'].shape[0]

        init_action = torch.randn(
            (B, self.config_dict["horizon"], self.config_dict["action_dim"]),
            device=self.device
        )

        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        timesteps = self.noise_scheduler.timesteps

        keep_action_all = record_config.get("action_all_target", False)
        action_all = [init_action.detach().cpu().numpy()] if keep_action_all else None

        result = {}
        probe_sample = None
        probe_timestep = None

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                net_out = self.diffusion_model(
                    sample=init_action,
                    timestep=t,
                    map_cond=map_cond,
                    env_cond=env
                )

                # only keep probe state if any extractor output is requested
                if (
                    record_config.get("attn_map", False)
                    or record_config.get("film_stats", False)
                    or record_config.get("grad_cam", False)
                ):
                    probe_sample = init_action.detach().clone()
                    probe_timestep = t

                init_action = self.noise_scheduler.step(
                    model_output=net_out,
                    timestep=t,
                    sample=init_action
                ).prev_sample

                if keep_action_all:
                    action_all.append(init_action.detach().cpu().numpy().copy())

        # only run extractor pass if needed
        need_extractors = (
            record_config.get("attn_map", False)
            or record_config.get("film_stats", False)
            or record_config.get("grad_cam", False)
        )

        if need_extractors:
            self.reset()
            self.diffusion_model.zero_grad(set_to_none=True)

            net_out = self.diffusion_model(
                sample=probe_sample,
                timestep=probe_timestep,
                map_cond=map_cond,
                env_cond=env
            )

            result = self.run_extractors(result, net_out, record_config)

        action_pred = init_action.detach().cpu().numpy()
        action_pred = self.unnormalize_data(action_pred)

        action_all_unnorm = None
        if keep_action_all:
            action_all_unnorm = np.stack(
                [self.unnormalize_data(step) for step in action_all],
                axis=0
            )  # [T+1, B, L, 2]

            action_all_unnorm = np.transpose(action_all_unnorm, (1, 0, 2, 3))  # [B, T+1, L, 2]
        gt_label = self.unnormalize_w_gt(gt_label.detach().cpu().numpy())

        return action_pred, action_all_unnorm, gt_label, result
    

    def collect_on_idx_range(self, range_of_idx, record_config=None):
        record_config = self._merge_record_config(record_config)
        buffers = self._init_output_buffers(record_config)

        batch_size = getattr(self.dataloader, "batch_size", 1)
        if batch_size is None:
            batch_size = 1
        dataset_len = len(self.dataloader.dataset)
        sample_indices = []

        for batch_idx in range_of_idx:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, dataset_len)
            sample_indices.extend(range(start_idx, end_idx))

        subset = torch.utils.data.Subset(self.dataloader.dataset, sample_indices)

        fast_dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(self.dataloader, "num_workers", 0),
            collate_fn=getattr(self.dataloader, "collate_fn", None),
            pin_memory=getattr(self.dataloader, "pin_memory", False),
        )
        print(f"total {len(fast_dataloader)} iterations for batch_size {batch_size}.")

        for actual_batch_idx, data in zip(range_of_idx, fast_dataloader):
            print(f"processing {actual_batch_idx}th samples")

            map_cpu = data["map"]
            env_cpu = data["env"]

            if record_config.get("grid_target", False):
                buffers["grid_target"].extend(map_cpu[:, 0, :, :])

            if record_config.get("st_target", False):
                buffers["st_target"].extend(self.unnormalize_data(env_cpu[:, :2]))

            if record_config.get("goal_target", False):
                buffers["goal_target"].extend(self.unnormalize_data(env_cpu[:, 2:]))

            data = self.move_tensor_dict(data, self.device)

            act_pred, act_all_unnorm, gt_label, result = self.collect_vis(data, record_config=record_config)

            if record_config.get("action_target", False):
                buffers["action_target"].extend(act_pred)

            if record_config.get("action_all_target", False):
                buffers["action_all_target"].extend(act_all_unnorm)

            if record_config.get("gt_target", False):
                buffers["gt_target"].extend(gt_label)

            if record_config.get("attn_map", False):
                # result["attention"] may already be batch-shaped; append carefully
                buffers["attn_map"].extend(result["attention"])

            if record_config.get("film_stats", False):
                buffers["film_stats"].append(result["film_stats"])

            if record_config.get("grad_cam", False):
                buffers["grad_cam"].extend(result["gradcam_1d"])

        return self._finalize_output_buffers(buffers)
        
    def visualize_action(self, action_target, grid_target, st_target, goal_target, num_vis, vis_fname, cols=5):
        """
            action_target: model's trajectory location prediction at time-steps. Expect numpy array of shape [B, 32, 2]
            grid_target: corresponding grid environment that trajectories lying on. Expect numpy array of shape [B, H, W]
            st_target: corresponding start point condition where trajectories should start from. Expect numpy array of shape [B, 2]
            goal_target: corresponding goal point condition where trajectories should end. Expect numpy array of shape [B, 2]
            num_vis: number of examples to visualize.
            vis_fname: visualization result file name
            cols: number of columns in output visualization file.

            Modified show_multiple function that colors paths based on collision status:
            - Red: collision detected
            - Blue: collision-free

            If ground truth action targets are given, you can visualize the ground-truth-examples!
        """

        indices = list(range(0, len(action_target), 1))
        rng = np.random.default_rng()
        # Randomly select num_vis samples for visualization
        vis_indices = sorted(rng.choice(indices, size=num_vis, replace=False))


        rows = (len(vis_indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        # Fix the axes handling
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        collision_count = 0
        success_count = 0
        status_list = []
        path_color_list = []
        goal_dist_list = []

        for i, idx in enumerate(indices):

            path = np.array(action_target[idx])
            grid = np.array(grid_target[idx])
            start = np.array(st_target[idx])
            goal = np.array(goal_target[idx])

            nx, ny = grid.shape[0], grid.shape[1]
            cell_size = 1.0
            bounds = [(0, nx * cell_size), (0, ny * cell_size)]
            origin = [bounds[0][0], bounds[1][0]]
            
            # Check collision status using your validation function
            is_collision_free = validate_path_collision_free(path, grid, cell_size, origin)
            
            # Check if goal is reached (within threshold)
            goal_distance = np.linalg.norm(path[-1] - goal)
            goal_reached = goal_distance < 0.5
            goal_dist_list.append(goal_distance)
            
            # Determine path color and status
            if is_collision_free and goal_reached:
                path_color = "blue"
                status = "SUCCESS"
                success_count += 1
            elif is_collision_free:
                path_color = "orange"
                status = "NO GOAL"
            else:
                path_color = "red"
                status = "COLLISION"
                collision_count += 1
            status_list.append(status)
            path_color_list.append(path_color)
            
        for i, vis_idx in enumerate(vis_indices):
            if i >= len(axes):
                break
            
            grid = np.array(grid_target[vis_idx])
            path = np.array(action_target[vis_idx])  
            start = np.array(st_target[vis_idx])
            goal = np.array(goal_target[vis_idx])

            nx, ny = grid.shape[0], grid.shape[1]
            cell_size = 1.0
            bounds = [(0, nx * cell_size), (0, ny * cell_size)]
            origin = [bounds[0][0], bounds[1][0]]

            status = status_list[vis_idx]
            path_color = path_color_list[vis_idx]

            ax = axes[i]
            ax.set_aspect("equal")
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_title(f"#{vis_idx}: {status}\nGoal dist: {goal_dist_list[vis_idx]:.2f}")
            
            # Draw obstacles
            xs = np.arange(nx) * cell_size + origin[0]
            ys = np.arange(ny) * cell_size + origin[1]
            for ix in range(nx):
                for iy in range(ny):
                    if grid[ix, iy]:
                        rect = plt.Rectangle(
                            (xs[ix], ys[iy]),
                            cell_size,
                            cell_size,
                            color="gray",
                            alpha=0.5,
                        )
                        ax.add_patch(rect)
            
            # Draw path with collision-based color
            ax.plot(path[:, 0], path[:, 1], color=path_color, linewidth=2, alpha=0.8)
            ax.scatter(path[:, 0], path[:, 1], color=path_color, s=15, alpha=0.6)
            
            # Draw start and goal
            ax.plot(start[0], start[1], "go", ms=8, label="Start")
            ax.plot(goal[0], goal[1], "ro", ms=8, label="Goal")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(vis_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Print statistics
        total_tests = len(indices)
        print(f"\n=== Test Results ===")
        print(f"Total tests: {total_tests}")
        print(f"Successful paths: {success_count}")
        print(f"Collision paths: {collision_count}")
        print(f"Success rate: {success_count/total_tests:.2%}")
        print(f"Collision rate: {collision_count/total_tests:.2%}")
        plt.savefig(f"{vis_fname}.pdf", dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'total_tests': total_tests,
            'success_count': success_count,
            'collision_count': collision_count,
            'success_rate': success_count/total_tests,
            'collision_rate': collision_count/total_tests
        }

    def visualize_action_process(self, action_all_target, grid_target, st_target=None, goal_target=None, vis_fname='diffusion_sample'):
        """
        action_all_target: [B, timesteps, 32, 2]
        grid_target: [B, W, H] (Based on your .T logic)
        st_target, goal_target: [B, 2]
        """
        B, T, N, _ = action_all_target.shape
        # Following your shape indexing: grid is [B, W, H]
        _, W, H = grid_target.shape
        
        for b in range(B):
            # Using float division for figsize to avoid rounding issues
            fig, ax = plt.subplots(figsize=(6, 6 * (H / W)), layout='constrained')
            
            # map_extent defines the boundaries of the image. 
            # [left, right, bottom, top]
            map_extent = [0, W, 0, H]

            # 1. Plot the static background
            # .T swaps W and H back to H, W for imshow's internal expectation
            ax.imshow(grid_target[b].T, cmap='Greys', origin='lower', 
                    vmin=0, vmax=1.5, extent=map_extent, zorder=1)

            # 2. Grid Setup
            # Using linspace ensures 0 and W/H are exactly hit
            tick_step=2
            ax.set_xticks(np.arange(0, W+1, tick_step))
            ax.set_yticks(np.arange(0, H+1, tick_step))

            # Minor ticks provide the grid lines for every single pixel
            ax.set_xticks(np.arange(0, W + 1, tick_step), minor=True)
            ax.set_yticks(np.arange(0, H + 1, tick_step), minor=True)
            
            # Draw grid lines on integer boundaries
            ax.grid(visible=True, which='minor', color='gray', linestyle='-', 
                linewidth=0.2, alpha=0.3, zorder=2)
            ax.grid(visible=True, which='major', color='gray', linestyle='-', 
                    linewidth=0.3, alpha=0.5, zorder=2)
            
            # 3. Initialize Trajectory
            line_plot, = ax.plot([], [], color='cyan', marker='o', markersize=2, 
                                linewidth=2, label='Denoising Trajectory', zorder=4)
            
            # 4. Start/Goal points
            # If map_extent is [0, W, 0, H], (0,0) is the bottom-left corner.
            if st_target is not None and goal_target is not None:
                ax.scatter(st_target[b, 0], st_target[b, 1], color='green', s=100, 
                        marker='o', label='Start', edgecolors='white', zorder=5)
                ax.scatter(goal_target[b, 0], goal_target[b, 1], color='red', s=150, 
                        marker='*', label='Goal', edgecolors='white', zorder=5)
            
            # CRITICAL: Match limits exactly to extent to avoid the 'shifted' look
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            
            ax.tick_params(axis='both', which='major', labelbottom=True, labelleft=True, labelsize=10)
            # Clean up labels but keep the grid
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            
            ax.set_title(f"Sample {b} | Initializing...")
            ax.legend(loc='upper right', fontsize='small')

            # plt.tight_layout()
            # fig.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9)

            def update(t):
                current_t = t # Standard forward-index animation
                
                # Using raw coordinates because extent=[0, W, 0, H] aligns 0 to the corner
                traj = action_all_target[b, current_t] 
                line_plot.set_data(traj[:, 0], traj[:, 1])
                
                ax.set_title(f"Sample {b} | Diffusion Timestep {current_t}")
                return [line_plot]

            ani = FuncAnimation(fig, update, frames=T, interval=100, blit=True)
            
            save_path = f"{vis_fname}_sampling_process_{b}.gif"
            ani.save(save_path, writer=PillowWriter(fps=10))
            plt.close(fig)
            
            print(f"Generated GIF: {save_path}")