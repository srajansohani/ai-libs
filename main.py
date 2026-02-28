from networks import NeuralNetwork
from layer import Layer
from matrix import Matrix

network = NeuralNetwork(3)

network.add_layer(2)
network.add_layer(2)
network.add_layer(1)

print(network.forward([1,1,1]).data)




network.printWeights()

network.backPropogation([0.5],[1,1,1],0.1)

network.printWeights()



