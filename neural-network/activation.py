import numpy as np;
class Activation:

    def __init__(self,name,value,derivative):
        self.name = name
        self.value = value
        self.derivative = derivative
        pass



class Activations:
    RELU = Activation("relu",lambda x: max(0,x), lambda y: 1 if y > 0 else 0)
    LINEAR = Activation("linear",lambda x: x, lambda y: 1)
    SIGMOID = Activation("sigmoid",lambda x : 1/(1 + pow(2.71828,-x)), lambda y: y * (1-y))
    TANH = Activation("tanh",lambda x: (2/(1 + pow(2.71828,-2*x))) - 1, lambda y: 1 - pow(y,2))

