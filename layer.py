from matrix import Matrix
from activation import Activations
import random
class Layer:
    def __init__(self, neurons,input_neurons, activation=Activations.LINEAR):
        self.neurons = neurons
        self.input_neurons = input_neurons
        self.values = Matrix(1,neurons)
        self.activation = activation
        self.weights = Matrix(input_neurons,neurons)
        self.bias = Matrix(1,neurons)
        self.initialize_weights()
        self.initialize_bias()
        pass

    def initialize_weights(self):
        for i in range(self.input_neurons):
            for j in range(self.neurons):
                limit = (1 / self.input_neurons) ** 0.5
                self.weights.data[i][j] = random.uniform(-limit, limit)
        
    def initialize_bias(self):
        for i in range(self.neurons):
            limit = (1 / self.neurons) ** 0.5
            self.bias.data[0][i] = random.uniform(-limit, limit)
    
    def forward(self, input):
        z = Matrix.add(Matrix.multiply(input,self.weights),self.bias)
        activated = Matrix.copy(z).apply_function(self.activation.value)
        self.values = activated
        return activated
    

    def backProp(self,delta,previous_layer_values,learning_rate = 0.01):
        

        #dL/dZ = dL/dA * delta
        delta =  Matrix.multiplyElementWise(delta,Matrix.copy(self.values).apply_function(self.activation.derivative))
        newdelta = Matrix.multiply(delta,self.weights.transpose())

        #dL/dW = a(l - 1) * ð(l)

        #dL/dW
        gradient = Matrix.multiply(previous_layer_values.transpose(),delta)

        #(learning_rate*dL/dW)
        gradient_descent = gradient.multiply_scalar(learning_rate)


        #gradient_descent algorithm
        # W = W - learning_rate * dL/dW
        self.weights = Matrix.subtract(self.weights,gradient_descent)

        bias_gradient = delta.multiply_scalar(learning_rate)
        self.bias = Matrix.subtract(self.bias, bias_gradient)
        return newdelta
    
    def printWeights(self):
        for i in range(self.input_neurons):
            for j in range(self.neurons):
                print(self.weights.data[i][j],end=" ")
            print()
        
        

    