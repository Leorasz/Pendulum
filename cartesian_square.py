import math
from tinygrad import Tensor
import matplotlib.pyplot as plt

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

epsilon = 0.01
    
full_range_hole1, full_range_hole2 = Tensor.arange(-math.pi/4, -math.pi/30, step=epsilon), Tensor.arange(math.pi/30, math.pi/4, step=epsilon)
full_range_donut_1D = full_range_hole1.cat(full_range_hole2)
full_range_donut_hole_1D = Tensor.arange(-math.pi/30, math.pi/30, step = epsilon)
donut = cartesian_donut(full_range_donut_1D, full_range_donut_hole_1D)

plt.scatter(donut[:, 0].numpy(), donut[:, 1].numpy())
plt.show()