from matrix import Matrix
from activation import Activations
import numpy as np
import random
class Layer:
    def __init__(self, neurons,input_neurons, activation=Activations.LINEAR):
        self.neurons = neurons
        self.input_neurons = input_neurons
        self.activation = activation
        self.bias = np.zeros((1,self.neurons))
        self.weights = np.ones((input_neurons,neurons))
        self.initialize_weights()
        self.initialize_bias()
        self.isCompiled = False
        pass

    def initialize_weights(self):
        for i in range(self.input_neurons):
            for j in range(self.neurons):
                limit = (1 / self.input_neurons) ** 0.5
                self.weights[i][j] = random.uniform(-limit, limit)
        
    def initialize_bias(self):
        for i in range(self.neurons):
            limit = (1 / self.neurons) ** 0.5
            self.bias[0][i] = random.uniform(-limit, limit)
    
    def forward(self, input):

        #Z = X * W  + b
        z = (np.dot(input,self.weights)) + self.bias

        #using numpy vectorization to apply activation function
        #a = f(z)
        activation_vector = np.vectorize(self.activation.value)
        activated = activation_vector(z)


        self.values = activated
        return activated
    

    def backProp(self,delta,previous_layer_values,learning_rate = 0.01):
        
        if(self.isCompiled == False):
            raise Exception("model is not compiled. Please compile the layer before backpropagation.")
        #dL/dZ = dL/dA * delta
        activation_derivative_vector = np.vectorize(self.activation.derivative)
        
        #dA/dZ = f'(Z)
        dA_dZ = activation_derivative_vector(self.values)
        delta = delta * dA_dZ
        
        
        # delta =  Matrix.multiplyElementWise(delta,Matrix.copy(self.values).apply_function(self.activation.derivative))
        # newdelta = Matrix.multiply(delta,self.weights.transpose())
        
        newdelta = np.dot(delta,self.weights.transpose())
        #dL/dW = a(l - 1) * ð(l)
        #dL/dW
        gradient = np.dot(previous_layer_values.transpose(),delta)

        #batch size as it is not necessary that now of elements in last batch will be same
        m = previous_layer_values.shape[0]

        #gradient_descent algorithm
        # W = W - learning_rate * (summation(dL/dW)) * (1/n)
        self.weights = self.weights -  gradient * learning_rate * (1/m)
        
        #for bias we will take average of delta across all samples in the batch
        db = np.sum(delta, axis=0, keepdims=True) * learning_rate * (1/m)
        # bias_gradient = delta * learning_rate
        self.bias = self.bias - db

        return newdelta
    
    def printWeights(self):
        for i in range(self.input_neurons):
            for j in range(self.neurons):
                print(self.weights[i][j],end=" ")
            print()
        
        
    def compile(self,optimizer,batch_size=256):
        
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.values = np.zeros((batch_size,self.neurons))
        self.isCompiled = True

    