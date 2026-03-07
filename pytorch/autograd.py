from torch import autograd
import torch

x = torch.tensor(3.0, requires_grad=True) # create a tensor with requires_grad=True to track computations

y = 3* x

z = 4* x

w = y + z

w.backward()
z.backward() # compute gradients

print(x.grad)

