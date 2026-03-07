import torch
from torch import autograd

class Dense:
    def __init__(self,neurons,input_neurons,activation):
        self.neurons = neurons
        self.input_neurons = input_neurons
        self.activation = activation
        self.values = None
        self.isCompiled = False
        self.initialize_weights()
        pass


    def compile(self,batch_size,learning_rate = 0.05):
        self.batch_size = batch_size
        self.isCompiled = True
        self.learning_rate = learning_rate
        self.bias = torch.zeros(1, self.neurons, requires_grad=True)

    def feed_forward(self,input):
        z = torch.matmul(input, self.weights) + self.bias
        return self.activation(z)

    def backward(self):
         with torch.no_grad():
            self.weights -= self.learning_rate * self.weights.grad
            self.bias -= self.learning_rate*self.bias.grad
        #  print("grad norm:", self.weights.grad.norm())
         self.weights.grad.zero_()
         self.bias.grad.zero_()
         
    
    def initialize_weights(self):
        self.weights = torch.randn(self.input_neurons, self.neurons) * (2 / self.input_neurons) ** 0.5
        self.weights.requires_grad_()
        pass