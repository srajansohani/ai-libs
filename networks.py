from layer import Layer
from matrix import Matrix

class NeuralNetwork:
    def __init__(self,input_neurons):
        self.layers = []
        self.input_neurons = input_neurons
        pass

    def add_layer(self,neurons,activation="linear"):
        if len(self.layers) == 0:
           layer = Layer(neurons,self.input_neurons)
        else:
            layer = Layer(neurons,self.layers[-1].neurons)

        self.layers.append(layer)
    

    def forward(self,input):
        input = Matrix(1,self.input_neurons,[input])
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    
    def backPropogation(self,lossEquivalence,input,learning_rate = 0.1):
        actualLossEquivalence = []
        for i in range(len(lossEquivalence)):
            actualLossEquivalence.append([lossEquivalence[i]])
        lossEquivalence = actualLossEquivalence
        lossEquivalence = Matrix(len(lossEquivalence),1,lossEquivalence)


        for i in range(len(self.layers)-1,-1,-1):
          if i > 0 :
            previous_layer_values = self.layers[i-1].values
            lossEquivalence = self.layers[i].backProp(lossEquivalence,previous_layer_values,learning_rate)

          else:
            previous_layer_values = Matrix.makeRow(input)
            lossEquivalence = self.layers[i].backProp(lossEquivalence,previous_layer_values,learning_rate)

    def printWeights(self):
        for i in range(len(self.layers)):
            print("Layer ",i+1)
            self.layers[i].printWeights()

    
