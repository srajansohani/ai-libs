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
        self.history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "layers": {}   # will be filled with indices when compile() is called
    }
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
       epochs = self.epochs
       total_samples = len(input_data)
       for epoch in range(epochs):
          total_loss = 0
          elements_covered = 0
          for batch_no,(input,expected_output) in enumerate(Utils.create_batches(input_data,expected_output_data,self.batch_size)):
                
                output = self.forward(input)

                batch_loss = self.loss.value(output,expected_output)

                total_loss += batch_loss*len(input)

                delta = self.loss.derivative(output,expected_output)

                self.backPropogation(delta,input,self.learning_rate)    

                elements_covered += len(input)
                Utils.show_progress(elements_covered,total_samples,batch_number=batch_no + 1)
          epoch_loss = total_loss / total_samples

          #For diagnostics and logging
          self.history["epochs"].append(epoch)
          self.history["train_loss"].append(epoch_loss)

          #Computing per layer diagnostics
          for idx,layer in enumerate(self.layers):
              #grad_norm using last_dW saved
              grad_norm = np.linalg.norm(layer.last_dW) if hasattr(layer, "last_dW") else 0
              #weight_norm after update
              weight_norm = np.linalg.norm(layer.weights)

              self.history["layers"][idx]["grad_norm"].append(grad_norm)
              self.history["layers"][idx]["weight_norm"].append(weight_norm)
          print(f"\nEpoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.6f}")


    
    def test(self,input_data,expected_output_data,log_to_history=True):
        elements_covered = 0
        total_samples = len(input_data)
        for batch_no, (input,expected_output) in enumerate(Utils.create_batches(input_data,expected_output_data,self.batch_size,shuffle=False)):
            output = self.forward(input)
            
            loss = self.loss.value(output,expected_output)

            total_loss = loss * len(input)

            elements_covered += len(input)
            print("testing ....")
            Utils.show_progress(elements_covered,total_samples,batch_number=batch_no + 1)
        
        test_loss = total_loss / total_samples
        print(f"\n Test Loss: {test_loss:.6f}")
        if log_to_history:
        # append val loss aligned with last epoch index (or None if no train done yet)
            self.history["val_loss"].append(test_loss)


    def compile(self,optimizer="adam",batch_size=32,learning_rate = 0.01,loss=Loss.MSE,epochs=1):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.epochs = epochs
        for idx, layer in enumerate(self.layers):
            layer.compile(optimizer, batch_size)
            # ensure history structure for layer diagnostics
            if idx not in self.history["layers"]:
                self.history["layers"][idx] = {
                    "weight_norm": [],
                    "grad_norm": []
                    # add more metrics here if you want
                }

    # Expected 2 d numpy arrays of X(input ) as (Dataponts,features) and y(expected output) as (Datapoints,output_neurons(expected features))
    def fit(self,X,Y,split_ratio=0.8):
       n = len(X)
       split = int(n * split_ratio)
       X_train, X_val = X[:split], X[split:]
       Y_train, Y_val = Y[:split], Y[split:]

       self.train(X_train, Y_train)
       val_loss = self.test(X_val, Y_val, log_to_history=True)
       return val_loss
       

 
