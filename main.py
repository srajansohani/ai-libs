import numpy as np
import time
# from activation import Activations
# from networks import NeuralNetwork
# from layer import Layer

# # arr = np.arange(1_000_000)

# # # Loop
# # t1 = time.time()
# # loop_res = [x * 2 for x in arr]
# # t2 = time.time()

# # # Vectorized
# # t3 = time.time()
# # vec_res = arr * 2
# # t4 = time.time()

# # print("Loop Time:", t2 - t1)
# # print("Vectorized Time:", t4 - t3)




# nn = NeuralNetwork(2)
# nn.add_layer(1,activation=Activations.RELU)

# nn.printWeights()

# nn.compile(batch_size=1,epochs=1,learning_rate=0.1)
# output = nn.forward(np.array([[1,2]]))
# print(output)



def create_batches(X, y, batch_size, shuffle=True,):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]

x = create_batches(np.array([[1,2],[3,4]]),np.array([[1],[2]]),1)
for value in x:
    print(value)