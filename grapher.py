import numpy as np
import matplotlib.pyplot as plt

# Example data for two trajectories
# Each trajectory is a sequence of (x, y) pairs over time
trajectory1 = np.array([[0, 0], [1, 1], [2, 3], [3, 4]])  # x, y for 4 time steps
trajectory2 = np.array([[0, 1], [1, 2], [2, 2], [3, 3], [4, 4]])  # x, y for 5 time steps

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Function to add time dimension
def add_time_dimension(trajectory):
    return np.column_stack([trajectory, np.arange(len(trajectory))])

# Convert 2D trajectories to 3D by adding time as the Z-axis
traj1_3d = add_time_dimension(trajectory1)
traj2_3d = add_time_dimension(trajectory2)

# Plot the trajectories
ax.plot(traj1_3d[:, 0], traj1_3d[:, 1], traj1_3d[:, 2], label='Trajectory 1')
ax.plot(traj2_3d[:, 0], traj2_3d[:, 1], traj2_3d[:, 2], label='Trajectory 2')

# Labeling
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend()

# Set view angle if needed
ax.view_init(elev=20., azim=35)

plt.show()
plt.clf()