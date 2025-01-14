import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn

Tensor.manual_seed(1337)
tau = 0.01
g = 9.8
m = l = 1
min_bound = -math.pi/4
max_bound = math.pi/4



def pendulum(t: Tensor, u: Tensor, w: Tensor) -> Tensor:
    assert t.shape[0] == u.shape[0] and u.shape == w.shape
    assert t.shape[2] == 2
    assert t.ndim == 3 and u.ndim == 2
    x1 = t[:, :, 0]
    x2 = t[:, :, 1]
    x1_next = x1 + tau*x2
    x2_next = x2 + tau*((g/l)*Tensor.sin(x1) + (u/(m*l**2))) + w
    raw_res = x1_next.stack(x2_next, dim=2)
    return Tensor.clip(raw_res, min_=min_bound, max_=max_bound)

def pendulum_small(t: Tensor, u: Tensor, w: Tensor) -> Tensor:
    assert t.shape[0] == u.shape[0] and u.shape == w.shape
    assert t.shape[1] == 2
    assert t.ndim == 2 and u.ndim == 1
    x1 = t[:, 0]
    x2 = t[:, 1]
    x1_next = x1 + tau*x2
    x2_next = x2 + tau*((g/l)*Tensor.sin(x1) + (u/(m*l**2))) + w
    raw_res = x1_next.stack(x2_next, dim=1)
    return Tensor.clip(raw_res, min_=min_bound, max_=max_bound)

def cartesian_prod(x: Tensor, y: Tensor) -> Tensor:
    assert x.ndim == 1 and y.ndim == 1
    one_d = x.repeat_interleave(y.shape[0])
    two_d = y.unsqueeze(1).repeat(1,x.shape[0]).T.flatten()
    return one_d.stack(two_d).T

def cartesian_square(x: Tensor) -> Tensor:
    return cartesian_prod(x, x)    

def cartesian_donut(donut_1D: Tensor, hole_1D: Tensor) -> Tensor:
    windows = cartesian_square(donut_1D)
    vertical_stripe = cartesian_prod(donut_1D, hole_1D)
    horizontal_stripe = cartesian_prod(hole_1D, donut_1D)
    return windows.cat(vertical_stripe).cat(horizontal_stripe)



#plt.scatter(full_range_hole[:, 0].numpy(), full_range_hole[:, 1].numpy())
#plt.show()
#plt.clf()
#exit()

class Model:
    def __init__(self):
        self.layers: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(2, 100), Tensor.relu,
            nn.Linear(100,100), Tensor.relu,
            nn.Linear(100,100), Tensor.relu,
            nn.Linear(100,1)
        ]

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers).squeeze(-1)

def get_combettes_pesquet_lipschitz(model: Model) -> float:
    weights = [layer.weight.numpy().T for layer in model.layers if isinstance(layer, nn.Linear)]
    test = weights[0]
    lip = 0

    for wi, weight_matrix in enumerate(weights[1:], start=1):
        test = np.matmul(test, weight_matrix)
        if wi != len(weights) - 1:
            test2 = weight_matrix
            for new_weight_matrix in weights[wi+1:]:
                test2 = np.matmul(test2, new_weight_matrix)
            eigenvalues2 = np.linalg.eigvals(np.matmul(test2.T,test2))
        else:
            eigenvalues2 = 1
        eigenvalues1 = np.linalg.eigvals(np.matmul(test.T,test))

        lip += np.sqrt(np.max(eigenvalues1)) * np.sqrt(np.max(eigenvalues2))

    return lip / np.power(2,len(weights)-1)

def get_trivial_lipschitz(model: Model) -> Tensor:
    spectrals = [layer.weight.flatten().max() for layer in model.layers if isinstance(layer, nn.Linear)]
    
    resses = [spectrals[0]]
    for i in spectrals[1:]:
        resses.append(resses[-1]*i)

    return resses[-1]

def save_model(model: Model, folder_name: str):
        linears = [linear for linear in model.layers if isinstance(linear, nn.Linear)]
        for li, linear in enumerate(linears):
            new_file_name = folder_name + "/" + str(li)
            np.save(new_file_name + "weights.npy", linear.weight.numpy())
            np.save(new_file_name + "biases.npy", linear.bias.numpy())

def load_model(model: Model, folder_name: str):
    linears = [linear for linear in model.layers if isinstance(linear, nn.Linear)]
    for li, linear in enumerate(linears):
        new_file_name = folder_name + "/" + str(li)
        linear.weight = Tensor(np.load(new_file_name + "weights.npy"))
        linear.bias = Tensor(np.load(new_file_name + "biases.npy"))



lx = 1.1
lu = 0.01
            
if __name__ == "__main__":
    p = 0.2
    delta = 2
    beta = 0.005/3
    betas = 0.045
    mhat = 10
    nhat = int(mhat/(delta*delta*betas))
    horizon = 20
    epsilon = 0.01
    std = 0.0

    eta = -0.001
    lamba = 80
    c = 0.1

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

    barrier = Model(); controller = Model()
    controller.layers.append(Tensor.sigmoid)

    #load_model(barrier, "lone_barrier_weights")
    load_model(barrier, "barrier_temp_weights")
    load_model(controller, "controller_temp_weights")
    
    barrier_lr = 3e-4; controller_lr = 3e-4

    barrier_opt = nn.optim.Adam(nn.state.get_parameters(barrier), lr=barrier_lr)
    controller_opt = nn.optim.Adam(nn.state.get_parameters(controller), lr=controller_lr)
    opt = nn.optim.OptimizerGroup(barrier_opt, controller_opt)

    @TinyJit
    @Tensor.train()
    def train_barrier() -> Tensor:
        barrier_opt.zero_grad()

        loss1 = -barrier(full_range) - eta #minimum limit on barrier
        loss2 = barrier(inits) - 1 - eta #initial states area
        loss3 = -barrier(unsafes) + lamba - eta #unsafe states area

        #loss6
        lb = get_trivial_lipschitz(barrier)
        loss6 = Tensor.relu(lb - 0.01)

        losses = []
        for loss in [loss1, loss2, loss3]:
            new_loss = Tensor.mean(Tensor.relu(loss))
            losses.append(new_loss)

        losses.append(loss6)
        
        final_loss = losses[0] + losses[1] + losses[2] + losses[3]
        final_loss.backward()

        barrier_opt.step()

        return final_loss, losses
        
    @TinyJit
    @Tensor.train()
    def train_both() -> Tensor:
        opt.zero_grad()

        loss1 = -barrier(full_range) - eta #minimum limit on barrier
        loss2 = barrier(inits) - 1 - eta #initial states area
        loss3 = -barrier(unsafes) + lamba - eta #unsafe states area

        #loss5
        controller_output = controller(full_range_hole)*10
        t = full_range_hole.unsqueeze(1).repeat(1, nhat, 1)
        u = controller_output.unsqueeze(-1).repeat(1, nhat)
        w = Tensor.normal(full_range_hole.shape[0], nhat, std=std)
        new_barrier = Tensor.mean(barrier(pendulum(t, u, w)), axis=1)
        loss5 = new_barrier - barrier(full_range_hole)# - c + delta - eta #controller
        loss5 = loss5 - 0.1 + delta - eta
        loss5 = Tensor.relu(loss5)

        #lipschitz loss
        lb = get_trivial_lipschitz(barrier)
        lg = get_trivial_lipschitz(controller)
        loss6 = Tensor.relu(lb*epsilon*(lx + lu*lg + 1) + epsilon - 0.1)

        losses = []
        for loss in [loss1, loss2, loss3, loss5]:
            new_loss = Tensor.mean(Tensor.relu(loss))
            losses.append(new_loss)

        losses.append(loss6)
        
        final_loss = losses[0] + losses[1] + losses[2] + losses[3] + losses[4]
        final_loss.backward()

        opt.step()

        return final_loss, losses

    counter = 0
    barrier_trained = True
    
    while True:
        if not barrier_trained:
            final_loss, losses = train_barrier()
            if (fl := final_loss.item()) == 0:
                barrier_trained = True
                print("Barrier trained!!")
                barrier_opt.zero_grad()
                save_model(barrier, "lone_barrier_weights")
            
            if counter % 1000 == 0:
                print(f"Total loss is {fl} for epoch {counter}")
                for li, loss in zip([1,2,3,6], losses):
                    print(f"Loss {li} is {loss.item()}")
                save_model(barrier, "barrier_temp_weights")

        else:
            final_loss, losses = train_both()
            if (fl := final_loss.item()) == 0:
                print("Success!!")
                break
            
            if counter % 100 == 0:
                print(f"Total loss is {fl} for epoch {counter}")
                for li, loss in zip([1,2,3,5,6], losses):
                    print(f"Loss {li} is {loss.item()}")
                save_model(barrier, "barrier_temp_weights")
                save_model(controller, "controller_temp_weights")

        counter += 1

    #save_model(barrier, "barrier_weights")
    #save_model(controller, "controller_weights")