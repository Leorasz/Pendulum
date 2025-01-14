from tinygrad import Tensor

a = Tensor([[1],[3],[5]])
b = Tensor([[7],[9],[11]])

c = a.stack(b, dim=2)

print(c.numpy())
print(c.shape)

