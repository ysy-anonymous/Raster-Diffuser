import numpy as np
import matplotlib.pyplot as plt

def upsample_2x2_to_8x8(heat_patch):
    return np.repeat(np.repeat(heat_patch, 4, axis=0), 4, axis=1)

def plot_attention_per_query(grid, traj, attn_bhql, start=None, goal=None, title_prefix=""):
    """
    grid: [8,8]
    traj: [32, 2]
    attn_bhql: [1,8,4] or [8,4]
    """
    grid = np.array(grid)
    traj = np.array(traj)
    attn = np.array(attn_bhql)

    if attn.ndim == 3:
        attn = attn[0]   # -> [8,4]

    num_queries = 8
    steps_per_query = len(traj) // num_queries

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    traj_plot = traj - 0.5
    if start is not None:
        start_plot = np.array(start) - 0.5
    if goal is not None:
        goal_plot = np.array(goal) - 0.5

    # Variable to store the heatmap image object for the colorbar
    im_heat = None 

    for q in range(num_queries):
        ax = axes[q // 4, q % 4]

        vec = attn[q]                    # [4]
        heat_patch = vec.reshape(2, 2)   # [2,2]
        heat = upsample_2x2_to_8x8(heat_patch)

        # Normalize attention between 0 and 1
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

        # Plot the base grid (map)
        ax.imshow(grid.T, cmap="Greys", origin="lower", vmin=0, vmax=1.5)
        
        # Plot the attention map WITHOUT the mask to restore the original blueish tint
        # Alpha is set to 0.6 so the red weights remain strong and visible
        im_heat = ax.imshow(heat.T, cmap="jet", origin="lower", alpha=0.6, vmin=0, vmax=1)

        # Grid lines and formatting
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.tick_params(which="minor", length=0)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)

        # Labels for the top legend
        label_traj = "Full Trajectory" if q == 0 else "_nolegend_"
        label_curr = f"Current Query ({steps_per_query} steps)" if q == 0 else "_nolegend_"
        label_start = "Start" if q == 0 else "_nolegend_"
        label_goal = "Goal" if q == 0 else "_nolegend_"

        # Plot trajectories and markers
        ax.plot(traj_plot[:, 0], traj_plot[:, 1], color="cyan", linewidth=1.5, alpha=0.7, label=label_traj)
        ax.scatter(traj_plot[:, 0], traj_plot[:, 1], color="cyan", s=20, alpha=0.7)

        start_idx = q * steps_per_query
        end_idx = start_idx + steps_per_query
        curr_points = traj_plot[start_idx:end_idx]

        ax.scatter(curr_points[:, 0], curr_points[:, 1], color="yellow", s=100, 
                   edgecolors="black", linewidths=1.0, label=label_curr, zorder=5)

        if start is not None:
            ax.scatter(start_plot[0], start_plot[1], c="green", s=70, marker="o", 
                       edgecolors="white", label=label_start, zorder=6)
        if goal is not None:
            ax.scatter(goal_plot[0], goal_plot[1], c="red", s=90, marker="*", 
                       edgecolors="white", label=label_goal, zorder=6)

        ax.set_title(f"{title_prefix} q={q} (steps {start_idx}-{end_idx-1})")

    # Shifted the legend x-anchor to 0.46 to perfectly center it over the subplots
    fig.legend(loc="upper center", ncol=4, fontsize="large", bbox_to_anchor=(0.46, 1.0))
    
    # Adjust layout to make room for the colorbar on the right and legend on top
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    
    # Add a unified colorbar on the right side of the figure for the attention weights
    cbar = fig.colorbar(im_heat, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label("Normalized Attention Weight", rotation=270, labelpad=20, fontsize=12, fontweight="bold")
    
    plt.show()


def plot_film_stats_bar(
    film_stats_list,
    idx=0,
    keys=("scale_mean_abs", "shift_mean_abs"),
    title="FiLM Stats",
    y_lim=(0, 10),
    layer_alias=None,
    rotation=45,
):
    item = film_stats_list[idx]

    module_names = list(item.keys())

    if layer_alias is None:
        layer_alias = {}

    display_names = [layer_alias.get(name, name) for name in module_names]

    x = np.arange(len(module_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, key in enumerate(keys):
        vals = [item[m].get(key, 0.0) for m in module_names]
        offset = (i - (len(keys) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=rotation, ha="right")

    ax.set_ylabel("Magnitude")
    ax.set_title(title)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.legend()
    plt.tight_layout()
    plt.show()


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def aggregate_film_raw_across_dataset(film_raw_list, layer_names):
    """
    film_raw_list: list of dicts
        each item:
        {
          layer_name: {
             "scale": [B, C],
             "shift": [B, C]
          },
          ...
        }

    Returns:
        aggregated = {
          layer_name: {
             "scale": np.ndarray [N_total_values],
             "shift": np.ndarray [N_total_values]
          },
          ...
        }
    """
    aggregated = {}

    for layer_name in layer_names:
        scale_all = []
        shift_all = []

        for item in film_raw_list:
            layer_item = item[layer_name]

            scale = _to_numpy(layer_item["scale"]).reshape(-1)
            shift = _to_numpy(layer_item["shift"]).reshape(-1)

            scale_all.append(scale)
            shift_all.append(shift)

        aggregated[layer_name] = {
            "scale": np.concatenate(scale_all, axis=0),
            "shift": np.concatenate(shift_all, axis=0),
        }

    return aggregated


def plot_film_distribution_compare(
    agg_base,
    agg_prop,
    layer_names,
    bins=80,
    density=True,
    xlim=None,
    save_path=None,
    title_prefix="FiLM raw distribution"
):
    """
    agg_base, agg_prop:
        output of aggregate_film_raw_across_dataset(...)

    Creates 2 x num_layers figure:
      row 1 = scale distribution
      row 2 = shift distribution
    """
    n_layers = len(layer_names)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 7), sharey="row")

    if n_layers == 1:
        axes = np.array(axes).reshape(2, 1)

    for j, layer_name in enumerate(layer_names):
        # scale
        ax = axes[0, j]
        base_scale = agg_base[layer_name]["scale"]
        prop_scale = agg_prop[layer_name]["scale"]

        ax.hist(base_scale, bins=bins, density=density, alpha=0.45, label="Diffuser")
        ax.hist(prop_scale, bins=bins, density=density, alpha=0.45, label="Raster-Diffuser")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(f"{layer_name}\nscale")
        ax.set_xlabel("Scale value")
        if j == 0:
            ax.set_ylabel("Density" if density else "Count")
        ax.legend()

        # shift
        ax = axes[1, j]
        base_shift = agg_base[layer_name]["shift"]
        prop_shift = agg_prop[layer_name]["shift"]

        ax.hist(base_shift, bins=bins, density=density, alpha=0.45, label="Diffuser")
        ax.hist(prop_shift, bins=bins, density=density, alpha=0.45, label="Raster-Diffuser")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(f"{layer_name}\nshift")
        ax.set_xlabel("Shift value")
        if j == 0:
            ax.set_ylabel("Density" if density else "Count")
        ax.legend()

    fig.suptitle(title_prefix, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def compute_layerwise_film_stats(aggregated_film):
    """
    aggregated_film:
        output of aggregate_film_raw_across_dataset(...)

    Returns:
        stats = {
          layer_name: {
            "scale_mean": ...,
            "scale_mean_abs": ...,
            "scale_std": ...,
            "shift_mean": ...,
            "shift_mean_abs": ...,
            "shift_std": ...,
          }
        }
    """
    stats = {}

    for layer_name, vals in aggregated_film.items():
        scale = vals["scale"]
        shift = vals["shift"]

        stats[layer_name] = {
            "scale_mean": float(np.mean(scale)),
            "scale_mean_abs": float(np.mean(np.abs(scale))),
            "scale_std": float(np.std(scale)),
            "shift_mean": float(np.mean(shift)),
            "shift_mean_abs": float(np.mean(np.abs(shift))),
            "shift_std": float(np.std(shift)),
        }

    return stats


def plot_layerwise_film_stats_compare(
    stats_base,
    stats_prop,
    layer_names,
    metrics=("scale_mean_abs", "shift_mean_abs", "scale_std", "shift_std"),
    layer_alias=None,
    figsize_per_subplot=(7, 5),
    rotation=30,
    save_path=None,
    title="Layer-wise FiLM statistics",
    use_horizontal=False,
):
    """
    stats_base, stats_prop:
        {
            layer_name: {
                "scale_mean": ...,
                "scale_mean_abs": ...,
                "scale_std": ...,
                "shift_mean": ...,
                "shift_mean_abs": ...,
                "shift_std": ...,
            }
        }

    layer_names:
        list of original layer names in plotting order

    layer_alias:
        dict mapping long layer names -> short display names
        example:
        {
            "diff_decoder.encoder.0.modulator.cond_encoder.2": "Enc0-Mod1",
            "diff_decoder.encoder.0.modulator2.cond_encoder.2": "Enc0-Mod2",
            ...
        }

    figsize_per_subplot:
        width, height per metric subplot
    """

    if layer_alias is None:
        layer_alias = {name: name for name in layer_names}

    short_names = [layer_alias.get(name, name) for name in layer_names]

    n_metrics = len(metrics)
    fig_w = figsize_per_subplot[0] * n_metrics
    fig_h = figsize_per_subplot[1]
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, fig_h))

    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(layer_names))
    width = 0.38

    for ax, metric in zip(axes, metrics):
        vals_base = [stats_base[layer][metric] for layer in layer_names]
        vals_prop = [stats_prop[layer][metric] for layer in layer_names]

        if use_horizontal:
            y = np.arange(len(layer_names))
            ax.barh(y + width / 2, vals_base, height=width, label="Baseline")
            ax.barh(y - width / 2, vals_prop, height=width, label="Proposed")
            ax.set_yticks(y)
            ax.set_yticklabels(short_names, fontsize=10)
            ax.set_title(metric, fontsize=13)
            ax.grid(axis="x", alpha=0.3)
        else:
            ax.bar(x - width / 2, vals_base, width, label="Baseline")
            ax.bar(x + width / 2, vals_prop, width, label="Proposed")
            ax.set_xticks(x)
            ax.set_xticklabels(short_names, rotation=rotation, ha="right", fontsize=10)
            ax.set_title(metric, fontsize=13)
            ax.grid(axis="y", alpha=0.3)

        ax.legend(fontsize=10)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()




from matplotlib.colors import ListedColormap, Normalize

def plot_gradcam_trajectory_aligned(
    grid,
    traj,
    cam,
    start=None,
    goal=None,
    title="Grad-CAM on trajectory",
    cell_size=1.0,
    bg_free="#f4f4f4",
    bg_obstacle="#7a7a7a",
    grid_color="#c8c8c8",
    path_cmap="jet",
    show_colorbar=True,
    show_ticks=True,
    tick_step=1,
    save_path=None,
    cam_mode="abs",          # "abs" or "signed"
    vmin=None,
    vmax=None,
):
    """
    cam_mode:
        - "abs": visualize |cam| with fixed [vmin, vmax], typically [0, global_abs_max]
        - "signed": visualize signed cam with fixed [vmin, vmax], typically [-global_abs_max, +global_abs_max]
    """

    grid = np.asarray(grid)
    traj = np.asarray(traj)
    cam = np.asarray(cam).squeeze()

    if cam.ndim != 1:
        raise ValueError(f"cam should become 1D after squeeze, got shape {cam.shape}")

    if len(cam) != len(traj):
        if len(cam) == len(traj) - 1:
            cam = np.concatenate([cam, cam[-1:]])
        else:
            raise ValueError(f"len(cam)={len(cam)} does not match len(traj)={len(traj)}")

    # choose visualization values
    if cam_mode == "abs":
        cam_vis = np.abs(cam)
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = float(cam_vis.max() + 1e-12)
    elif cam_mode == "signed":
        cam_vis = cam
        if vmax is None:
            vmax = float(np.abs(cam).max() + 1e-12)
        if vmin is None:
            vmin = -vmax
    else:
        raise ValueError("cam_mode must be 'abs' or 'signed'")

    W, H = grid.shape[0], grid.shape[1]
    map_extent = [0, W * cell_size, 0, H * cell_size]

    cmap_bg = ListedColormap([bg_free, bg_obstacle])

    fig, ax = plt.subplots(figsize=(6, 6 * (H / max(W, 1))), layout="constrained")

    ax.imshow(
        grid.T,
        cmap=cmap_bg,
        origin="lower",
        vmin=0,
        vmax=1,
        extent=map_extent,
        interpolation="nearest",
        zorder=1,
    )

    ax.set_xticks(np.arange(0, W * cell_size + 1e-9, tick_step))
    ax.set_yticks(np.arange(0, H * cell_size + 1e-9, tick_step))
    ax.set_xticks(np.arange(0, W * cell_size + 1e-9, cell_size), minor=True)
    ax.set_yticks(np.arange(0, H * cell_size + 1e-9, cell_size), minor=True)

    ax.grid(which="minor", color=grid_color, linewidth=0.5, alpha=0.55, zorder=2)
    ax.grid(which="major", color=grid_color, linewidth=0.8, alpha=0.75, zorder=2)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(path_cmap)

    for i in range(len(traj) - 1):
        xs = [traj[i, 0], traj[i + 1, 0]]
        ys = [traj[i, 1], traj[i + 1, 1]]
        ax.plot(
            xs,
            ys,
            linewidth=3.0,
            alpha=0.95,
            color=cmap(norm(cam_vis[i])),
            solid_capstyle="round",
            zorder=4,
        )

    sc = ax.scatter(
        traj[:, 0],
        traj[:, 1],
        c=cam_vis,
        cmap=path_cmap,
        norm=norm,
        s=34,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )

    if start is not None:
        start = np.asarray(start)
        ax.scatter(
            start[0], start[1],
            color="#2ca02c",
            s=110,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            zorder=6,
            label="Start"
        )

    if goal is not None:
        goal = np.asarray(goal)
        ax.scatter(
            goal[0], goal[1],
            color="#d62728",
            s=170,
            marker="*",
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
            label="Goal"
        )

    ax.set_xlim(0, W * cell_size)
    ax.set_ylim(0, H * cell_size)
    ax.set_aspect("equal")

    if not show_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", which="both", length=0)
    else:
        ax.tick_params(axis="both", which="major", labelsize=10)

    ax.set_title(title)

    if (start is not None) or (goal is not None):
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    if show_colorbar:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Grad-CAM intensity", rotation=90)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def compare_gradcam_trajectory(out_base, out_prop, idx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (name, out) in zip(axes, [("Baseline", out_base), ("Proposed", out_prop)]):
        grid = np.array(out["grid_target"][idx])
        traj = np.array(out["action_target"][idx])
        cam = np.array(out["grad_cam"][idx]).squeeze()
        st = np.array(out["st_target"][idx])
        goal = np.array(out["goal_target"][idx])

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        ax.imshow(grid.T, cmap="Greys", origin="lower", vmin=0, vmax=1.5)

        for i in range(len(traj) - 1):
            xs = [traj[i, 0], traj[i + 1, 0]]
            ys = [traj[i, 1], traj[i + 1, 1]]
            ax.plot(xs, ys, linewidth=3, alpha=0.9, color=plt.cm.jet(cam[i]))

        ax.scatter(traj[:, 0], traj[:, 1], c=cam, cmap="jet", s=25, edgecolors="black", linewidths=0.2)
        ax.scatter(st[0], st[1], c="green", s=80, marker="o", edgecolors="white")
        ax.scatter(goal[0], goal[1], c="red", s=100, marker="*", edgecolors="white")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()