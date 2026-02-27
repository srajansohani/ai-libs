from matrix import Matrix
import random
class Layer:
    def __init__(self, neurons,input_neurons, activation="linear"):
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
                self.weights.data[i][j] = random.uniform(-1,1)
        
    def initialize_bias(self):
        for i in range(self.neurons):
            self.bias.data[0][i] = random.uniform(-1,1)
    
    def forward(self, input):
        result = Matrix.add(Matrix.multiply(input,self.weights),self.bias)
        self.values = result
        return result
    

    def backProp(self,lossEquivalence,previous_layer_values,learning_rate = 0.1):
 
        #dL/dW
        gradient = Matrix.multiply(lossEquivalence,previous_layer_values).transpose()

        print(gradient.data)

        #learning_rate*dL/dW
        gradient_descent = gradient.multiply_scalar(learning_rate)

        #gradient_descent algorithm
        # W = W - learning_rate * dL/dW
        print(gradient_descent.data)
        self.weights = Matrix.subtract(self.weights,gradient_descent)

        newLossEquivalence = Matrix.multiply(self.weights,lossEquivalence)

        return newLossEquivalence
    
    def printWeights(self):
        for i in range(self.input_neurons):
            for j in range(self.neurons):
                print(self.weights.data[i][j],end=" ")
            print()
        
        

    