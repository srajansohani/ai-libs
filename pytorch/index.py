import torch


x = torch.zeros(2,3) # create a tensor of size 2x3 filled with zeros

print(x)

x = torch.tensor([[5]],dtype=torch.float32,requires_grad=True) # create a tensor from a list with specified data type

w1 = torch.ones(1,3) # create a tensor of size 2x3 filled with ones

w2 = torch.randn(3,4)

w3 = torch.randn(4,1) # create a tensor of size 3x4 filled with random numbers from normal distribution


z1 = torch.matmul(x,w1) # matrix multiplication of x and w1

z2 = torch.matmul(z1,w2) # matrix multiplication of z1 and w2

y = torch.matmul(z2,w3) # matrix multiplication of z2 and w3

y_hat = torch.tensor([[1.0]],dtype=torch.float32) # create a tensor of size 1x1 filled with 1.0

k = y - y_hat # compute the difference between predicted and actual values

k.backward() # compute gradients

print(x.grad)


