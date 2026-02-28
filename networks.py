from layer import Layer
from matrix import Matrix
from utils import Utils
from LossFunctions import Loss
import numpy as np
import sys
import time
from activation import Activations
class NeuralNetwork:
    def __init__(self,input_neurons):
        self.layers = []
        self.input_neurons = input_neurons
        self.learning_rate = 0.05
        pass

    def add_layer(self,neurons,activation=Activations.SIGMOID):
        if len(self.layers) == 0:
           layer = Layer(neurons,self.input_neurons,activation=activation)
        else:
            layer = Layer(neurons,self.layers[-1].neurons,activation=activation)

        self.layers.append(layer)
    

    def forward(self,input):
        input = Matrix(1,self.input_neurons,[input])
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    
    def backPropogation(self,delta,input,learning_rate = 0.1):
        delta = Matrix.makeRow(delta)
        for i in range(len(self.layers)-1,-1,-1):
          if i > 0 :
            previous_layer_values = self.layers[i-1].values
            delta = self.layers[i].backProp(delta,previous_layer_values,learning_rate)

          else:
            previous_layer_values = Matrix.makeRow(input)
            delta = self.layers[i].backProp(delta,previous_layer_values,learning_rate)

    def printWeights(self):
        for i in range(len(self.layers)):
            print("Layer ",i+1)
            self.layers[i].printWeights()


    #Curent implementation stochastic gradient descent
    
    def train(self,input_data,expected_output_data,loss=Loss.MSE,epochs=1):
       elements_covered = 0
       correct = 0
       incorrect = 0
       for j in range(epochs):
          total_loss = []
          indices = np.random.permutation(len(input_data))
          preds = []
          true_values = []
          for i in indices:
           input = input_data[i]
           expected_output = expected_output_data[i]
           output = self.forward(input)
           lossEquivalence = loss.derivative(output.data[0],expected_output)
           preds.append(output.data[0])
           true_values.append(expected_output)
           self.backPropogation(lossEquivalence,input,self.learning_rate)
           elements_covered += 1
           Utils.show_progress(elements_covered, len(input_data)*epochs)
           time.sleep(0.02)
          print("\n Training completed. Average Loss: ", loss.value(np.array(preds),np.array(true_values)))
      
       
    def test(self,input_data,expected_output_data,loss=Loss.MSE):
       elements_covered = 0
       preds = []
       for i in range(len(input_data)):
            input = input_data[i]
            expected_output = expected_output_data[i]
            output = self.forward(input)
            print("Test - 1","output: ",output.data," expected output: ",expected_output)
            preds.append(output.data[0])
            elements_covered += 1
            Utils.show_progress(elements_covered, len(input_data))
            time.sleep(0.02)
       print("\n Testing completed. Average Loss: ", loss.value(preds,expected_output)/(len(input_data)))


