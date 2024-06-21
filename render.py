import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

import torch

TYPE_TO_COLOR = {
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}
step_stride = 3

with open("gnn_rollout_data.pkl", 'rb') as file:
    rollout_data = pickle.load(file)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plot_info = []
for ax_i, (label, rollout_field) in enumerate(
    [("Ground truth", "ground_truth_rollout"),
    ("Prediction", "predicted_rollout")]):
    
    trajectory = rollout_data[rollout_field]

    ax = axes[ax_i]
    ax.set_title(label)
    bounds = [[0.1, 0.9], [0.1, 0.9]] # rollout_data["metadata"]["bounds"]
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.)
    points = {
        particle_type: ax.plot([], [], "o", ms=2, color=color)[0]
        for particle_type, color in TYPE_TO_COLOR.items()}
    plot_info.append((ax, trajectory, points))

    num_steps = trajectory.shape[0]
    def update(step_i):
        outputs = []
        for _, trajectory, points in plot_info:
            for particle_type, line in points.items():
                mask = rollout_data["particle_types"] == particle_type
                line.set_data(trajectory[step_i, mask, 0],
                            trajectory[step_i, mask, 1])
                outputs.append(line)
        return outputs
        
    anim = animation.FuncAnimation(
        fig, update,
        frames=np.arange(0, num_steps, step_stride), interval=10)

    # Save the animation
    anim.save('animation.gif', writer='pillow', fps=30)

    plt.show(block=True)