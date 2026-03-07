import numpy as np
class LossFunction:
    def __init__(self,name,value,derivative):
        self.name = name,
        self.value = value
        self.derivative = derivative

    
class Loss:

    MSE = LossFunction(
        "Mean Squared Error",
        lambda y_pred, y_true: ((y_pred - y_true) ** 2).mean(),
        lambda y_pred, y_true: 2 * (y_pred - y_true) / y_true.size
    )

    MAE = LossFunction(
        "Mean Absolute Error",
        lambda y_pred, y_true: np.abs(y_pred - y_true).mean(),
        lambda y_pred, y_true: np.sign(y_pred - y_true) / y_true.size
    )

    BCE = LossFunction(
        "Binary Cross Entropy",
        lambda y_pred, y_true: -(
            y_true * np.log(y_pred + 1e-9) +
            (1 - y_true) * np.log(1 - y_pred + 1e-9)
        ).mean(),
        lambda y_pred, y_true: -(
            y_true / (y_pred + 1e-9) -
            (1 - y_true) / (1 - y_pred + 1e-9)
        ) / y_true.size
    )