import torch
from layer import Dense

from utils import Utils

class NN:
    def __init__(self,input_neurons):
        self.layers = []
        self.input_neurons = input_neurons
        self.isCompiled = False
    
    def add_layer(self,neurons,activation):
        if len(self.layers) == 0:
            layer = Dense(neurons,self.input_neurons,activation)
        else:
            layer = Dense(neurons,self.layers[-1].neurons,activation) 
        self.layers.append(layer)
    
    def forward(self,x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x
    
    def backward(self,output,expected_output):
        loss = self.loss(output,expected_output)
        loss.backward()

        for layer in self.layers:
            layer.backward() ## To update weights
    
    def compile(self,batch_size = 256,epochs = 10,loss=None,learning_rate = 0.01):
        self.batch_size = batch_size
        self.loss = loss
        self.epochs = epochs
        self.learning_rate = learning_rate
        for layer in self.layers:
            layer.compile(batch_size,learning_rate)
    

    def train(self,train_data_input,train_data_expected_output):
        for epoch in range(self.epochs):
            elements_covered = 0
            total_samples = len(train_data_input)
            loss  = 0
            num_batches = 0
            for batch_no, (input,expected_output) in enumerate(Utils.create_batches(train_data_input,train_data_expected_output,self.batch_size)):
                output = self.forward(input)
                loss += self.loss(output,expected_output).item()
                self.backward(output,expected_output)
                elements_covered += input.shape[0]
                num_batches +=1
                Utils.show_progress(elements_covered,total_samples,batch_number=batch_no + 1)
            print("Loss in epoch ",epoch+1," is ",loss/num_batches)
                
            

