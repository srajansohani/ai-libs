import torch
import math
class Activations:
    RELU =  lambda x: torch.relu(x)
    LINEAR = lambda x : x
    SIGMOID = lambda x: (1/(1 + math.exp(-x)))
    TANH = lambda x: (2/(1 + math.exp(-2*x))) - 1



class Loss:
    MSE = lambda y_pred, y_true: ((y_pred - y_true) ** 2).mean()