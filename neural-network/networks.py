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
        pass

    def add_layer(self,neurons,activation=Activations.SIGMOID):
        if len(self.layers) == 0:
           layer = Layer(neurons,self.input_neurons,activation=activation)
        else:
            layer = Layer(neurons,self.layers[-1].neurons,activation=activation)

        self.layers.append(layer)
    

    def forward(self,input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    
    def backPropogation(self,delta,input,learning_rate = 0.1):
        for i in range(len(self.layers)-1,-1,-1):
          if i > 0 :
            previous_layer_values = self.layers[i-1].values
            delta = self.layers[i].backProp(delta,previous_layer_values,learning_rate)

          else:
            previous_layer_values = input
            delta = self.layers[i].backProp(delta,previous_layer_values,learning_rate)

    def printWeights(self):
        print(self.layers)
        for i in range(len(self.layers)):
            print("Layer ",i+1)
            self.layers[i].printWeights()


    #Curent implementation stochastic gradient descent
    
    def train(self,input_data,expected_output_data):
       elements_covered = 0
       correct = 0
       incorrect = 0
       epochs = self.epochs
       total_samples = len(input_data)
       for epoch in range(epochs):
          total_loss = 0
          elements_covered = 0
          for batch_no,(input,expected_output) in enumerate(Utils.create_batches(input_data,expected_output_data,self.batch_size)):
                
                output = self.forward(input)

                loss = self.loss.value(output,expected_output)

                total_loss += loss*len(input)

                delta = self.loss.derivative(output,expected_output)

                self.backPropogation(delta,input,self.learning_rate)    

                elements_covered += len(input)
                Utils.show_progress(elements_covered,total_samples,batch_number=batch_no + 1)
          total_loss = total_loss / total_samples
          print(f"\nEpoch {epoch+1}/{self.epochs} - Loss: {total_loss:.6f}")

    
    def test(self,input_data,expected_output_data):
        elements_covered = 0
        total_samples = len(input_data)
        for batch_no, (input,expected_output) in enumerate(Utils.create_batches(input_data,expected_output_data,self.batch_size,shuffle=False)):
            output = self.forward(input)
            
            loss = self.loss.value(output,expected_output)

            total_loss = loss * len(input)

            elements_covered += len(input)
            print("testing ....")
            Utils.show_progress(elements_covered,total_samples,batch_number=batch_no + 1)
        
        print(f"\n Test Loss: {total_loss/total_samples:.6f}")


    def compile(self,optimizer="adam",batch_size=32,learning_rate = 0.01,loss=Loss.MSE,epochs=1):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.epochs = epochs
        for layer in self.layers:
            layer.compile(optimizer,batch_size)
        pass

    # Expected 2 d numpy arrays of X(input ) as (Dataponts,features) and y(expected output) as (Datapoints,output_neurons(expected features))
    def fit(self,X,Y,split_ratio=0.8):
        
       X_train, X_test = X[:int(len(X)*split_ratio)]
       Y_train, Y_test = Y[:int(len(Y)*split_ratio)]

       self.train(X_train,Y_train)

       self.test(X_test,Y_test)
       

 
