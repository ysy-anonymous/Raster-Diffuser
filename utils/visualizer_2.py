import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Pdfpages for Visualization
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import os

from core.diffuser.datasets.plane_dataset_embeed import PlanePlanningDataSets
from utils.load_utils import build_noise_scheduler_from_config
from utils.dataset_utils import validate_path_collision_free

class Visualizer():
    def __init__(self, diffusion_model, config_dict, test_datapath, device, visual_config=None):
        self.diffusion_model = diffusion_model
        self.config_dict = config_dict
        self.stats = self.move_np_dict(config_dict['normalizer'])
        self.dataset = PlanePlanningDataSets(dataset_path=test_datapath, **config_dict['dataset'])
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False
        )
        self.gt_norm = self.dataset.norm_stats

        self.noise_scheduler = build_noise_scheduler_from_config(config_dict)
        if hasattr(self.diffusion_model, 'set_noise_scheduler'):
            self.diffusion_model.set_noise_scheduler(self.noise_scheduler)

        self.device = device
        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()

        # -------------------------------
        # Visualization configuration
        # -------------------------------
        self.tick_step = 2

        self.coord_font_size = 16
        self.title_font_size = 14
        self.legend_font_size = 10

        self.start_marker_size = 140
        self.goal_marker_size = 260
        self.marker_edge_width = 1.5

        self.trajectory_linewidth = 2.5
        self.trajectory_marker_size = 2

        self.obstacle_face_color = "royalblue"
        self.obstacle_edge_color = "navy"
        self.obstacle_alpha = 0.65
        self.obstacle_linewidth = 0.4

        # Optional override from constructor
        if visual_config is not None:
            for key, value in visual_config.items():
                if not hasattr(self, key):
                    raise KeyError(f"Unknown visualization config key: {key}")
                setattr(self, key, value)

    def set_visualization_config(self, **kwargs):
        """
        Update visualization parameters after initialization.

        Example:
            visualizer.set_visualization_config(
                coord_font_size=20,
                start_marker_size=180,
                goal_marker_size=320,
                obstacle_face_color="red"
            )
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"Unknown visualization config key: {key}")
            setattr(self, key, value)

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

    # unnormalize trajectory data
    def unnormalize_data(self, ndata):
        ndata = (ndata+1)/2
        data = ndata * (np.array(self.stats['max']) - np.array(self.stats['min'])) + np.array(self.stats['min'])
        return data
    
    def unnormalize_w_gt(self, ndata):
        ndata = (ndata+1)/2
        data = ndata * (np.array(self.gt_norm['max']) - np.array(self.gt_norm['min'])) + np.array(self.gt_norm['min'])
        return data
    
    # normalize trajectory data
    def normalize_data(self, ndata):
        n, T, d = ndata.shape  
        flat = ndata.reshape(-1, d) 

        eps = 1e-8
        norm = (flat - self.stats['min']) / (self.stats['max'] - self.stats['min'] + eps)  # → [0, 1]
        norm = norm * 2 - 1  # → [-1, 1]

        norm = norm.reshape(n, T, d)
        return norm
    

    def setup_grid_coordinates(
        self,
        ax,
        W,
        H,
        tick_step=None,
        coord_font_size=None,
        show_axis=True,
    ):
        """
        Make plot coordinates follow grid coordinates instead of pixel-center coordinates.

        Grid coordinate convention:
            x-axis: 0, 1, ..., W
            y-axis: 0, 1, ..., H
        """
        if tick_step is None:
            tick_step = self.tick_step

        if coord_font_size is None:
            coord_font_size = self.coord_font_size

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(0, W + 1, tick_step))
        ax.set_yticks(np.arange(0, H + 1, tick_step))

        ax.set_xticks(np.arange(0, W + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, H + 1, 1), minor=True)

        ax.grid(
            visible=True,
            which="minor",
            color="gray",
            linestyle="-",
            linewidth=0.2,
            alpha=0.3,
            zorder=2,
        )
        ax.grid(
            visible=True,
            which="major",
            color="gray",
            linestyle="-",
            linewidth=0.3,
            alpha=0.5,
            zorder=2,
        )

        if show_axis:
            ax.tick_params(
                axis="both",
                which="major",
                labelbottom=True,
                labelleft=True,
                labelsize=coord_font_size,
            )
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="both", which="both", length=0)

    def collect_vis(self, data):
    
        gt_label = data['sample'] # grid sample that is true
        map = data['map']; env = data['env'] # map, env condition each
        init_action = torch.randn((1, self.config_dict['horizon'], self.config_dict['action_dim']), device=self.device)
        
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        timesteps = self.noise_scheduler.timesteps

        print("noise scheduler timestep: ", len(timesteps))
        
        action_all = [init_action.detach().cpu().numpy()[0]]
        raster_all = [] # (Timestep, B, num_iters, Hs, Ws)
        for i, t in enumerate(timesteps, 0):
            with torch.no_grad():
                net_out, raster_vis = self.diffusion_model( # rasterized image for diffusion timesteps
                    sample=init_action,
                    timestep=t,
                    map_cond=map,
                    env_cond=env
                )
                noise_pred = net_out
                
                init_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=init_action
                ).prev_sample
                
                action_all.append(init_action.detach().cpu().numpy()[0].copy())
                raster_all.append(raster_vis) # (Timestep, B, num_iters, Hs, Ws)

                if i == 0: # only first raster_vis image for visualization
                    raster_init = raster_vis
        
        raster_all = np.array(raster_all).transpose(1, 0, 2, 3, 4) # (B, Timestep, num_iters, Hs, Ws)
        action_pred = init_action.detach().cpu().numpy()[0]
        action_pred = self.unnormalize_data(action_pred)

        raster_img = raster_vis # (B, num_iters, Hs, Ws)
            
        action_all_unnorm = []
        for action_step in action_all:
            action_step_unnorm = self.unnormalize_data(action_step)
            action_all_unnorm.append(action_step_unnorm)
        gt_label = self.unnormalize_w_gt(gt_label.detach().cpu().numpy())

        return action_pred, action_all_unnorm, raster_init, raster_img, raster_all, gt_label # remove batch dim and add!


    def collect_on_idx_range(self, range_of_idx, collect_only_input_data=False):
        
        action_target = []
        action_all_target = []
        raster_init_target = []
        raster_target = []
        raster_all_target = []

        gt_target = [] # ground truth trajectory
        grid_target = [] # map (grid without distance map) visualization
        st_target = [] # start point
        goal_target = [] # goal point

        # 1. Safely extract original dataloader configurations
        batch_size = getattr(self.dataloader, 'batch_size', 1)
        if batch_size is None:
            batch_size = 1
        dataset_len = len(self.dataloader.dataset)
        sample_indices = []

        # 2. Map the requested batch iterations to the exact sample indices
        for batch_idx in range_of_idx:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, dataset_len)
            sample_indices.extend(range(start_idx, end_idx))
        
        # 3. Create a subset containing only the requested data
        subset = torch.utils.data.Subset(self.dataloader.dataset, sample_indices)

        # 4.Spin up a temporary, highly targeted dataloader
        fast_dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False, # Force sequential reading for this specific range
            num_workers=getattr(self.dataloader, "num_workers", 0),
            collate_fn=getattr(self.dataloader, "collate_fn", None),
            pin_memory=getattr(self.dataloader, "pin_memory", False)
        )

        # 5. Iterate ONLY over the targeted dataloader
        for actual_batch_idx, data in zip(range_of_idx, fast_dataloader):
            print(f"processing {actual_batch_idx}th samples")

            map = data['map']; env = data['env']
            grid_target.extend(map[:, 0, :, :])
            st_target.extend(self.unnormalize_data(env[:, :2]))
            goal_target.extend(self.unnormalize_data(env[:, 2:]))

            data = self.move_tensor_dict(data, self.device)
            if not collect_only_input_data:
                act_pred, act_all_unnorm, raster_init, raster_img, raster_img_all, gt_label = self.collect_vis(data)
                action_target.append(act_pred)
                action_all_target.append(act_all_unnorm)
                raster_init_target.extend(raster_init)
                raster_target.extend(raster_img)
                raster_all_target.extend(raster_img_all)
                gt_target.extend(gt_label)
            else:
                gt_target.extend(data['sample'].detach().cpu().numpy())

        if not collect_only_input_data:
            out_dict = {
                "action_target": np.array(action_target),
                "action_all_target": np.array(action_all_target),
                "raster_init_target": np.array(raster_init_target),
                "raster_target": np.array(raster_target),
                "raster_all_target": np.array(raster_all_target),
                "gt_target": np.array(gt_target),
                "grid_target": np.array(grid_target),
                "st_target": np.array(st_target),
                "goal_target": np.array(goal_target),
            }
        else:
            out_dict = {
                "action_target": None,
                "action_all_target": None,
                "raster_init_target": None,
                "raster_target": None,
                "raster_all_target": None,
                "gt_target": gt_target,
                "grid_target": grid_target,
                "st_target": st_target,
                "goal_target": goal_target,
            }

        return out_dict # output dictionaries
    

    def visualize_action(self, action_target, grid_target, st_target, goal_target, num_vis, vis_fname, cols=5):
        """
            action_target: model's trajectory location prediction at time-steps. Expect numpy array of shape [B, 32, 2]
            grid_target: corresponding grid environment that trajectories lying on. Expect numpy array of shape [B, H, W]
            st_target: corresponding start point condition where trajectories should start from. Expect numpy array of shape [B, 2]
            goal_target: corresponding goal point condition where trajectories should end. Expect numpy array of shape [B, 2]
            num_vis: number of examples to visualize.
            vis_fname: visualization result file name
            cols: number of columns in output visualization file.
        """

        coord_font_size = self.coord_font_size
        title_font_size = self.title_font_size
        legend_font_size = self.legend_font_size

        tick_step = 2

        # Extra spacing to avoid overlap between subplot title and coordinate labels
        title_pad = 12

        indices = list(range(0, len(action_target), 1))
        rng = np.random.default_rng()

        vis_indices = sorted(rng.choice(indices, size=num_vis, replace=False))

        rows = (len(vis_indices) + cols - 1) // cols

        # Make figure slightly taller to give room for titles and the shared legend
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 3.2, rows * 3.45 + 0.8),
            constrained_layout=False,
        )

        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = np.array(axes)
        else:
            axes = axes.flatten()

        # More vertical space between rows prevents title/tick overlap
        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.08,
            top=0.78,
            wspace=0.35,
            hspace=0.65,
        )

        collision_count = 0
        success_count = 0
        status_list = []
        path_color_list = []
        goal_dist_list = []

        for idx in indices:
            path = np.array(action_target[idx])
            grid = np.array(grid_target[idx])
            goal = np.array(goal_target[idx])

            nx, ny = grid.shape[0], grid.shape[1]
            cell_size = 1.0
            bounds = [(0, nx * cell_size), (0, ny * cell_size)]
            origin = [bounds[0][0], bounds[1][0]]

            is_collision_free = validate_path_collision_free(path, grid, cell_size, origin)

            goal_distance = np.linalg.norm(path[-1] - goal)
            goal_reached = goal_distance < 0.5
            goal_dist_list.append(goal_distance)

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

        legend_handles = None
        legend_labels = None

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

            # Explicit tick step = 2
            ax.set_xticks(np.arange(0, nx + 1, tick_step))
            ax.set_yticks(np.arange(0, ny + 1, tick_step))

            # Minor ticks every grid cell
            ax.set_xticks(np.arange(0, nx + 1, 1), minor=True)
            ax.set_yticks(np.arange(0, ny + 1, 1), minor=True)

            ax.grid(
                visible=True,
                which="minor",
                color="gray",
                linestyle="-",
                linewidth=0.2,
                alpha=0.3,
                zorder=2,
            )
            ax.grid(
                visible=True,
                which="major",
                color="gray",
                linestyle="-",
                linewidth=0.3,
                alpha=0.5,
                zorder=2,
            )

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=coord_font_size,
                pad=2,
            )

            # Larger pad prevents title from touching the top coordinate labels
            ax.set_title(
                f"#{vis_idx}: {status}\nGoal dist: {goal_dist_list[vis_idx]:.2f}",
                fontsize=title_font_size,
                pad=title_pad,
            )

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
                            zorder=1,
                        )
                        ax.add_patch(rect)

            ax.plot(
                path[:, 0],
                path[:, 1],
                color=path_color,
                linewidth=2,
                alpha=0.8,
                label="Trajectory" if i == 0 else None,
                zorder=4,
            )

            ax.scatter(
                path[:, 0],
                path[:, 1],
                color=path_color,
                s=15,
                alpha=0.6,
                zorder=4,
            )

            ax.scatter(
                start[0],
                start[1],
                color="green",
                s=self.start_marker_size,
                edgecolors="white",
                linewidths=self.marker_edge_width,
                zorder=5,
                label="Start" if i == 0 else None,
            )

            ax.scatter(
                goal[0],
                goal[1],
                color="red",
                marker="*",
                s=self.goal_marker_size,
                edgecolors="white",
                linewidths=self.marker_edge_width,
                zorder=5,
                label="Goal" if i == 0 else None,
            )

            if i == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        for i in range(len(vis_indices), len(axes)):
            axes[i].set_visible(False)

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.96),
                ncol=len(legend_labels),
                frameon=True,
                fontsize=legend_font_size,
            )

        total_tests = len(indices)
        print(f"\n=== Test Results ===")
        print(f"Total tests: {total_tests}")
        print(f"Successful paths: {success_count}")
        print(f"Collision paths: {collision_count}")
        print(f"Success rate: {success_count / total_tests:.2%}")
        print(f"Collision rate: {collision_count / total_tests:.2%}")

        plt.savefig(
            f"{vis_fname}.pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.show()

        return {
            "total_tests": total_tests,
            "success_count": success_count,
            "collision_count": collision_count,
            "success_rate": success_count / total_tests,
            "collision_rate": collision_count / total_tests,
        }
    
    def visualize_raster(
        self,
        raster_img,
        action_target=None,
        st_target=None,
        goal_target=None,
        vis_fname=None,
        grid_target=None,
    ):
        """
            raster_img: rasterized image of trajectory. Expected numpy array of shape [B, num_iters, H, W].
            action_target: corresponding target trajectory. If given as None, this will be excluded in visualization.
                Expected numpy array of shape [B, 32, 2]
            st_target: corresponding start point location. If one of st_target, goal_target is None, those will be excluded.
                Expected numpy array of shape [B, 2]
            goal_target: corresponding goal point location. Expected numpy array of shape [B, 2]
            grid_target: optional obstacle map. If None, obstacle visualization is skipped.
                Expected shape is either [B, W, H] or [B, H, W]
        """
        B, num_iters, H, W = raster_img.shape

        def draw_obstacle_grid(ax, grid, W, H, zorder=3):
            grid = np.asarray(grid)

            if grid.shape == (W, H):
                for ix in range(W):
                    for iy in range(H):
                        if grid[ix, iy]:
                            rect = plt.Rectangle(
                                (ix, iy),
                                1.0,
                                1.0,
                                facecolor=self.obstacle_face_color,
                                edgecolor=self.obstacle_edge_color,
                                linewidth=self.obstacle_linewidth,
                                alpha=self.obstacle_alpha,
                                zorder=zorder,
                            )
                            ax.add_patch(rect)

            elif grid.shape == (H, W):
                for iy in range(H):
                    for ix in range(W):
                        if grid[iy, ix]:
                            rect = plt.Rectangle(
                                (ix, iy),
                                1.0,
                                1.0,
                                facecolor=self.obstacle_face_color,
                                edgecolor=self.obstacle_edge_color,
                                linewidth=self.obstacle_linewidth,
                                alpha=self.obstacle_alpha,
                                zorder=zorder,
                            )
                            ax.add_patch(rect)

            else:
                raise ValueError(
                    f"grid_target sample has shape {grid.shape}, "
                    f"but expected either (W, H)=({W}, {H}) or (H, W)=({H}, {W})."
                )

        if vis_fname is None:
            pdf_filename = "raster_visualization.pdf"
        else:
            pdf_filename = vis_fname

        with PdfPages(pdf_filename) as pdf:
            for b in range(B):
                # Use manual spacing control so we can reserve room for a shared top legend
                fig, axes = plt.subplots(
                    1,
                    num_iters,
                    figsize=(num_iters * 4, 4 * (H / W) + 0.8),
                    constrained_layout=False,
                )

                if num_iters == 1:
                    axes = [axes]

                # Leave space at the top for the shared legend
                fig.subplots_adjust(top=0.82, wspace=0.25)

                map_extent = [0, W, 0, H]

                legend_handles = None
                legend_labels = None

                for i in range(num_iters):
                    ax = axes[i]

                    # Raster background
                    ax.imshow(
                        raster_img[b, i],
                        cmap="Greys",
                        origin="lower",
                        vmin=0,
                        vmax=1.2,
                        extent=map_extent,
                        zorder=1,
                    )

                    # Optional obstacle overlay
                    if grid_target is not None:
                        draw_obstacle_grid(
                            ax=ax,
                            grid=grid_target[b],
                            W=W,
                            H=H,
                            zorder=3,
                        )

                    # Use grid coordinates, not pixel-center coordinates
                    self.setup_grid_coordinates(
                        ax=ax,
                        W=W,
                        H=H,
                        show_axis=True,
                    )

                    # Trajectory
                    if action_target is not None:
                        traj = action_target[b]
                        ax.plot(
                            traj[:, 0],
                            traj[:, 1],
                            color="cyan",
                            linewidth=self.trajectory_linewidth,
                            label="Trajectory" if i == 0 else None,
                            zorder=4,
                        )

                    # Start and goal
                    if st_target is not None and goal_target is not None:
                        st = st_target[b]
                        go = goal_target[b]

                        ax.scatter(
                            st[0],
                            st[1],
                            color="green",
                            s=self.start_marker_size,
                            label="Start" if i == 0 else None,
                            edgecolors="white",
                            linewidths=self.marker_edge_width,
                            zorder=5,
                        )
                        ax.scatter(
                            go[0],
                            go[1],
                            color="red",
                            marker="*",
                            s=self.goal_marker_size,
                            label="Goal" if i == 0 else None,
                            edgecolors="white",
                            linewidths=self.marker_edge_width,
                            zorder=5,
                        )

                    ax.set_title(
                        f"Sample {b} | Iter {i}",
                        fontsize=self.title_font_size,
                        pad=8,
                    )

                    # Collect legend handles from the first subplot only
                    if i == 0:
                        legend_handles, legend_labels = ax.get_legend_handles_labels()

                # IMPORTANT:
                # Do NOT call ax.legend(...) anywhere.
                # Use one shared legend for the whole figure.
                if legend_handles:
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.98),
                        ncol=len(legend_labels),
                        frameon=True,
                        fontsize=self.legend_font_size,
                    )

                pdf.savefig(
                    fig,
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.08,
                )
                plt.close(fig)

        print(f"Visualization saved to {pdf_filename}")

    def visualize_action_process(
        self,
        action_all_target,
        grid_target,
        st_target=None,
        goal_target=None,
        vis_fname="diffusion_sample",
    ):
        """
        action_all_target: [B, timesteps, 32, 2]
        grid_target: [B, W, H]
        st_target, goal_target: [B, 2]
        """
        B, T, N, _ = action_all_target.shape
        _, W, H = grid_target.shape

        coord_font_size = self.coord_font_size
        title_font_size = self.title_font_size
        legend_font_size = self.legend_font_size

        start_marker_size = self.start_marker_size
        goal_marker_size = self.goal_marker_size
        marker_edge_width = self.marker_edge_width

        for b in range(B):
            # Use manual layout so we can reserve space for the shared top legend
            fig, ax = plt.subplots(
                figsize=(6, 6 * (H / W) + 0.8),
                constrained_layout=False,
            )

            # Reserve top area for legend
            fig.subplots_adjust(top=0.82)

            map_extent = [0, W, 0, H]

            ax.imshow(
                grid_target[b].T,
                cmap="Greys",
                origin="lower",
                vmin=0,
                vmax=1.5,
                extent=map_extent,
                zorder=1,
            )

            self.setup_grid_coordinates(
                ax=ax,
                W=W,
                H=H,
                tick_step=2,
                coord_font_size=coord_font_size,
                show_axis=True,
            )

            line_plot, = ax.plot(
                [],
                [],
                color="cyan",
                marker="o",
                markersize=2,
                linewidth=2.5,
                label="Denoising Trajectory",
                zorder=4,
            )

            if st_target is not None and goal_target is not None:
                ax.scatter(
                    st_target[b, 0],
                    st_target[b, 1],
                    color="green",
                    s=start_marker_size,
                    marker="o",
                    label="Start",
                    edgecolors="white",
                    linewidths=marker_edge_width,
                    zorder=5,
                )
                ax.scatter(
                    goal_target[b, 0],
                    goal_target[b, 1],
                    color="red",
                    s=goal_marker_size,
                    marker="*",
                    label="Goal",
                    edgecolors="white",
                    linewidths=marker_edge_width,
                    zorder=5,
                )

            ax.set_title(
                f"Sample {b} | Initializing...",
                fontsize=title_font_size,
                pad=8,
            )

            # One shared legend above the figure
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.98),
                    ncol=len(labels),
                    frameon=True,
                    fontsize=legend_font_size,
                )

            def update(t):
                traj = action_all_target[b, t]
                line_plot.set_data(traj[:, 0], traj[:, 1])

                ax.set_title(
                    f"Sample {b} | Diffusion Timestep {t}",
                    fontsize=title_font_size,
                    pad=8,
                )

                return [line_plot]

            ani = FuncAnimation(fig, update, frames=T, interval=100, blit=True)

            save_path = f"{vis_fname}_sampling_process_{b}.gif"
            ani.save(save_path, writer=PillowWriter(fps=10))
            plt.close(fig)

            print(f"Generated GIF: {save_path}")
    
    def visualize_raster_process(
        self,
        raster_img,
        action_all_target=None,
        st_target=None,
        goal_target=None,
        vis_fname="diffusion_sample_w_raster",
        grid_target=None,
    ):
        """
            raster_img: rasterized images of trajectory for diffusion timestep.
                Expected numpy array of shape [B, timesteps, num_iters, H, W].
            action_all_target: corresponding target trajectory for diffusion timestep.
                Expected shape [B, timesteps, 32, 2]
            st_target: corresponding start point location.
                Expected numpy array of shape [B, 2]
            goal_target: corresponding goal point location.
                Expected numpy array of shape [B, 2]
            grid_target: optional obstacle map. If None, obstacle visualization is skipped.
                Expected shape is either [B, W, H] or [B, H, W]
        """
        B, T, num_iters, H, W = raster_img.shape

        coord_font_size = self.coord_font_size
        title_font_size = self.title_font_size
        legend_font_size = self.legend_font_size

        start_marker_size = self.start_marker_size
        goal_marker_size = self.goal_marker_size
        marker_edge_width = self.marker_edge_width

        obstacle_face_color = self.obstacle_face_color
        obstacle_edge_color = self.obstacle_edge_color
        obstacle_alpha = self.obstacle_alpha
        obstacle_linewidth = self.obstacle_linewidth

        if action_all_target is not None:
            action_all_target = action_all_target[:, 1:, :, :]
            anim_frames = min(T, action_all_target.shape[1])
        else:
            anim_frames = T

        def draw_obstacle_grid(ax, grid, W, H, zorder=3):
            grid = np.asarray(grid)

            if grid.shape == (W, H):
                for ix in range(W):
                    for iy in range(H):
                        if grid[ix, iy]:
                            rect = plt.Rectangle(
                                (ix, iy),
                                1.0,
                                1.0,
                                facecolor=obstacle_face_color,
                                edgecolor=obstacle_edge_color,
                                linewidth=obstacle_linewidth,
                                alpha=obstacle_alpha,
                                zorder=zorder,
                            )
                            ax.add_patch(rect)

            elif grid.shape == (H, W):
                for iy in range(H):
                    for ix in range(W):
                        if grid[iy, ix]:
                            rect = plt.Rectangle(
                                (ix, iy),
                                1.0,
                                1.0,
                                facecolor=obstacle_face_color,
                                edgecolor=obstacle_edge_color,
                                linewidth=obstacle_linewidth,
                                alpha=obstacle_alpha,
                                zorder=zorder,
                            )
                            ax.add_patch(rect)

            else:
                raise ValueError(
                    f"grid_target sample has shape {grid.shape}, "
                    f"but expected either (W, H)=({W}, {H}) or (H, W)=({H}, {W})."
                )

        for b in range(B):
            # Use manual layout so we can reserve space for the shared top legend
            fig, ax = plt.subplots(
                figsize=(6.5, 6 * (H / W) + 0.8),
                constrained_layout=False,
            )

            # Reserve top area for legend
            fig.subplots_adjust(top=0.82)

            map_extent = [0, W, 0, H]

            img_plot = ax.imshow(
                raster_img[b, 0, -1],
                cmap="Greys",
                origin="lower",
                vmin=0,
                vmax=1.2,
                extent=map_extent,
                zorder=1,
            )

            if grid_target is not None:
                draw_obstacle_grid(
                    ax=ax,
                    grid=grid_target[b],
                    W=W,
                    H=H,
                    zorder=3,
                )

            self.setup_grid_coordinates(
                ax=ax,
                W=W,
                H=H,
                tick_step=2,
                coord_font_size=coord_font_size,
                show_axis=True,
            )

            if action_all_target is not None:
                line_plot, = ax.plot(
                    [],
                    [],
                    color="cyan",
                    linewidth=2.5,
                    label="Trajectory",
                    zorder=4,
                )
            else:
                line_plot = None

            if st_target is not None and goal_target is not None:
                ax.scatter(
                    st_target[b, 0],
                    st_target[b, 1],
                    color="green",
                    s=start_marker_size,
                    label="Start",
                    edgecolors="white",
                    linewidths=marker_edge_width,
                    zorder=5,
                )
                ax.scatter(
                    goal_target[b, 0],
                    goal_target[b, 1],
                    color="red",
                    marker="*",
                    s=goal_marker_size,
                    label="Goal",
                    edgecolors="white",
                    linewidths=marker_edge_width,
                    zorder=5,
                )

            ax.set_title(
                f"Sample {b} | Timestep 0",
                fontsize=title_font_size,
                pad=8,
            )

            # One shared legend above the figure
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.98),
                    ncol=len(labels),
                    frameon=True,
                    fontsize=legend_font_size,
                )

            def update(t):
                img_plot.set_data(raster_img[b, t, -1])

                if action_all_target is not None:
                    traj = action_all_target[b, t]
                    line_plot.set_data(traj[:, 0], traj[:, 1])

                ax.set_title(
                    f"Sample {b} | Diffusion Timestep {t}",
                    fontsize=title_font_size,
                    pad=8,
                )

                return [img_plot, line_plot] if line_plot else [img_plot]

            ani = FuncAnimation(
                fig,
                update,
                frames=anim_frames,
                interval=100,
                blit=True,
            )

            save_path = f"{vis_fname}_sample_{b}.gif"
            ani.save(save_path, writer=PillowWriter(fps=10))
            plt.close(fig)

            print(f"Generated Raster Process GIF: {save_path}")