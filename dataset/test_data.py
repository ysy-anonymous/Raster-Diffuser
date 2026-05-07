import numpy as np
import matplotlib.pyplot as plt

data = np.load("/exhdd/seungyu/diffusion_motion/dataset/train_data_set_flatten.npy", allow_pickle=True)
training_data = data.item() 

starts = np.array(training_data["start"])
goals = np.array(training_data["goal"])
obstacles_list = training_data["map"]  


idx = 12  
start = starts[idx]
goal = goals[idx]
obstacles = obstacles_list[idx] 


def visualize_training_sample(start, goal, grid, bounds, cell_size, save_path='/exhdd/seungyu/diffusion_motion/vis/single_sample_vis.png'):
    nx, ny = grid.shape
    origin = np.array([b[0] for b in bounds])
    
    xs = np.arange(nx) * cell_size + origin[0]
    ys = np.arange(ny) * cell_size + origin[1]
    
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title(f"Training Sample (start={start}, goal={goal})")


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

    ax.plot(start[0], start[1], "go", ms=8, label="start")
    ax.plot(goal[0], goal[1], "ro", ms=8, label="goal")
    ax.legend()
    plt.xticks(np.arange(bounds[0][0], bounds[0][1]+1, cell_size))
    plt.yticks(np.arange(bounds[1][0], bounds[1][1]+1, cell_size))
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

bounds = [(0.0, 10.0), (0.0, 10.0)]
cell_size = 1.0

save_path = '/exhdd/seungyu/diffusion_motion/vis/single_sample_vis.png'
visualize_training_sample(start, goal, obstacles, bounds, cell_size, save_path=save_path)