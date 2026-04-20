import torch
from torch import autograd

class RNNLayer:
    def __init__(self,neurons,input_neurons,time_steps,activation):
        self.neurons = neurons
        self.input_neurons = input_neurons
        self.activation = activation
        self.input_time_steps = time_steps
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
        #z = ot-1 * Whh + xt * Wxh + b
        curr_hidden_state = torch.rand(input.shape[0], self.neurons)
        hidden_states = torch.rand(input.shape[0],input.shape[1], self.neurons)
        for t in range(input.shape[1]):
            x_t = input[:,t,:]
            z = torch.matmul(curr_hidden_state, self.weights_hh) + torch.matmul(x_t,self.weights) + self.bias
            curr_hidden_state = self.activation(z)
            hidden_states[:,t,:] = curr_hidden_state
       
        return self.activation(z),hidden_states

    def backward(self):
         with torch.no_grad():
            self.weights -= self.learning_rate * self.weights.grad
            self.weights_hh -= self.learning_rate * self.weights_hh.grad
            self.bias -= self.learning_rate*self.bias.grad
        #  print("grad norm:", self.weights.grad.norm())
         self.weights.grad.zero_()
         self.weights_hh.grad.zero_()
         self.bias.grad.zero_()
         
    
    def initialize_weights(self):
        self.weights = torch.randn(self.input_neurons, self.neurons) * (2 / self.input_neurons) ** 0.5
        self.weights.requires_grad_()

        self.weights_hh = torch.randn(self.neurons, self.neurons) * (2 / self.neurons) ** 0.5
        self.weights_hh.requires_grad_()
        pass



