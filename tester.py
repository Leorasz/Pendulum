import numpy as np
import math
from tinygrad import Tensor, TinyJit
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
from pendulum import pendulum, pendulum_small, load_model, Model, cartesian_square, cartesian_donut, get_trivial_lipschitz, get_combettes_pesquet_lipschitz

barrier = Model()
controller = Model()

load_model(barrier, "barrier_temp_weights")
load_model(controller, "controller_temp_weights")
controller.layers.append(Tensor.sigmoid)

p = 0.2
delta = 2
beta = 0.005/3
betas = 0.045
mhat = 10
nhat = int(mhat/(delta*delta*betas))
horizon = 20
epsilon = 0.01
std = 0.0

eta = -0.01
c = 0.74
lamba = 1

lx = 1.1
lu = 0.01

nhat = int(mhat/(delta*delta*betas))
batch_size = int(100_0000/nhat)

full_range_1D = Tensor.arange(-math.pi/4, math.pi/4, step=epsilon)
full_range = cartesian_square(full_range_1D)

inits_1D = Tensor.arange(-math.pi/15, math.pi/15, step=epsilon)
inits = cartesian_square(inits_1D)

unsafes1, unsafes2 = Tensor.arange(-math.pi/4, -math.pi/6, step=epsilon), Tensor.arange(math.pi/6, math.pi/4, step=epsilon)
unsafes_donut_1D = unsafes1.cat(unsafes2)
unsafes_hole_1D = Tensor.arange(-math.pi/6, math.pi/6, step = epsilon)
unsafes = cartesian_donut(unsafes_donut_1D, unsafes_hole_1D)

full_range_hole1, full_range_hole2 = Tensor.arange(-math.pi/4, -math.pi/30, step=epsilon), Tensor.arange(math.pi/30, math.pi/4, step=epsilon)
full_range_donut_1D = full_range_hole1.cat(full_range_hole2)
full_range_donut_hole_1D = Tensor.arange(-math.pi/30, math.pi/30, step=epsilon)
full_range_hole = cartesian_donut(full_range_donut_1D, full_range_donut_hole_1D)

def print_cp_lipschitz(barrier: Model, controller: Model):
    lb = get_combettes_pesquet_lipschitz(barrier)
    lg = get_combettes_pesquet_lipschitz(controller)
    print(f"Lipschitz for barrier is {lb} and lipschitz for controller is {lg}")
    print(f"Lipschitz value for system is {lb*(lx+lu*lg+1)}")
    lip_value = lb*epsilon*(lx + lu*lg + 1) + eta
    if lip_value > 0:
        print(f"Lipschitz value is not good, it's {lip_value}")
    else:
        print("Lipschitz value is good")

def print_trivial_lipschitz(barrier: Model, controller: Model):
    lb = get_trivial_lipschitz(barrier).item()
    lg = get_trivial_lipschitz(controller).item()
    l = lb*(lx+lu*lg+1)
    print(f"Lipschitz for barrier is {lb} and lipschitz for controller is {lg}")
    print(f"Lipschitz value for system is {l}")
    lip_value = l*epsilon + eta
    if lip_value > 0:
        print(f"Lipschitz value is not good, it's {lip_value}")
    else:
        print("Lipschitz value is good")
    print(f"epsilon needs to be less than {-eta/l}")

def graph_barrier(barrier: Model):
    full_range_1D = Tensor.arange(-math.pi/4, math.pi/4, step=0.01)
    full_range = cartesian_square(full_range_1D)
    # Generate data for the plane
    X = Y = full_range_1D.numpy()
    X, Y = np.meshgrid(X, Y)
    Z = barrier(full_range).reshape(full_range_1D.shape[0],-1).numpy()

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with color representing Z
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

    # Customize the plot
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Barrier Value')
    ax.set_title('Barrier Function')

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Barrier Value')

    # Optionally, adjust the view angle
    ax.view_init(elev=20., azim=35)

    # Show the plot
    plt.show()
    plt.clf()

@TinyJit
@Tensor.test()
def graph_loss5(barrier: Model, controller: Model):
    function_epsilon = 0.05
    full_range_1D = Tensor.arange(-math.pi/6, math.pi/6, step=function_epsilon)
    full_range = cartesian_square(full_range_1D)

    nhat = 10

    print("gonna find loss5")
    controller_output = controller(full_range)*10
    t = full_range.unsqueeze(1).repeat(1, nhat, 1)
    u = controller_output.unsqueeze(-1).repeat(1, nhat)
    w = Tensor.normal(full_range.shape[0], nhat, std=std)
    new_barrier = Tensor.mean(barrier(pendulum(t, u, w)), axis=1)
    loss5 = new_barrier - barrier(full_range) - c + delta - eta #controller

    print("gonna make meshgrid")
    positions = velocities = full_range_1D.numpy()
    X, Y = np.meshgrid(positions, velocities)
    Z = loss5.reshape(full_range_1D.shape[0], -1).numpy()

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X,Y,Z, cmap='viridis', linewidth=0, antialiased=False)

    # plane_X, plane_Y = np.meshgrid(np.linspace(positions.min(), positions.max(), 100), 
    #                             np.linspace(velocities.min(), velocities.max(), 100))
    # plane_Z = np.zeros_like(plane_X)  # Z = 0 everywhere

    # # Plot the plane
    # plane = ax.plot_surface(plane_X, plane_Y, plane_Z, alpha=0.5, color='red')

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Loss 5')
    ax.set_title('Graph of Loss 5')

    fig.colorbar(surf, shrink=0.4, aspect=5, label='Loss 5')

    ax.view_init(elev=20., azim=35)

    plt.show()
    plt.clf()

@TinyJit
@Tensor.test()
def graph_controller(controller: Model, time_steps: int, std=std):
    function_epsilon = 0.1
    full_range_1D = Tensor.arange(-math.pi/15, math.pi/15, step=function_epsilon)
    full_range = cartesian_square(full_range_1D)

    trajectories = [full_range]

    for _ in range(time_steps):
        controller_output = controller(trajectories[-1])*10
        w = Tensor.normal(full_range.shape[0], std=std)
        new_state = pendulum_small(trajectories[-1], controller_output, w)
        trajectories.append(new_state)

    trajectories_tens = trajectories[0].stack(*trajectories[1:])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(trajectories_tens.shape[1]):
        raw_path = trajectories_tens[:, i, :].T
        inverted_time = time_steps+1 - Tensor.arange(time_steps+1).reshape(1,-1)
        path = raw_path.cat(inverted_time, dim=0).numpy()
        ax.plot(path[0], path[1], path[2])

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Time")

    ax.view_init(elev=20., azim=35)

    plt.show()


@TinyJit
@Tensor.test()
def find_list_variance(tensors: List[Tensor]) -> float:
    means = [t.mean().item() for t in tensors]
    mean = sum(means)/len(means)

    squares = [(t - mean).square().mean().item() for t in tensors]
    variance = sum(squares)/len(squares)

    return variance

@TinyJit
@Tensor.test()
def test_main_losses(barrier: Model, controller: Model):

    loss1 = -barrier(full_range) - eta #minimum limit on barrier
    loss2 = barrier(inits) - 1 - eta #initial states area
    loss3 = -barrier(unsafes) + lamba - eta #unsafe states area

    #loss5
    batches = full_range_hole.split(batch_size)
    loss5s = []
    new_barriers = []
    for batch in tqdm(batches):
        controller_output = controller(batch)*10
        t = batch.unsqueeze(1).repeat(1, nhat, 1)
        u = controller_output.unsqueeze(-1).repeat(1, nhat)
        w = Tensor.normal(batch.shape[0], nhat, std=std)
        new_barrier = Tensor.mean(barrier(pendulum(t, u, w)), axis=1)
        new_barriers.append(new_barrier)
        loss5u = new_barrier - barrier(batch) - c + delta - eta #controller
        print(f"this max is {Tensor.max(loss5u.flatten()).item()}")
        loss5s.append(Tensor.mean(Tensor.relu(loss5u)).item())

    losses = []
    for loss in [loss1, loss2, loss3]:
        new_loss = Tensor.mean(Tensor.relu(loss))
        losses.append(new_loss.item())

    losses.append(sum(loss5s)/len(loss5s))

    variance = find_list_variance(new_barriers)

    return variance, losses    

@TinyJit
@Tensor.test()
def test_system(barrier: Model, controller: Model):
    #main losses
    print("Main losses")
    variance, losses = test_main_losses(barrier, controller)
    for li, loss in zip([1,2,3,5], losses):
        if loss > 0:
            print(f"Loss {li} is {loss}, not good")
        else:
            print(f"Loss {li} is good")
    #loss4
    loss4 = (1 + c*horizon)/p - lamba - eta
    if loss4 > 0:
        print(f"Loss 4 is {loss4}, not good")
    else:
        print("Loss 4 looks good")
    print(f"c needs to be less than {(p*(lamba + eta) - 1)/horizon}")
    #lipschitz continuity
    print("trivial lipschitz")
    print_trivial_lipschitz(barrier, controller)
    #mhat is right
    if variance > mhat:
        print(f"Variance is not good, greater than mhat, mhat is {mhat} while variance is {variance}")
    else:
        print("Variance is good")
    #make sure barrier looks right
    print("Graphing barrier")
    graph_barrier(barrier)
    print("Graphing controller")
    graph_controller(controller, horizon, std=std)

test_system(barrier, controller)
graph_loss5(barrier, controller)
# graph_controller(controller, 20, std=0)
