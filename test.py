import pandas as pd
import numpy as np
import os
from networks import NeuralNetwork
from activation import Activations

admission_Data = pd.read_csv("../datasets/admission_data.csv")

normalized_admission_data = (admission_Data - admission_Data.min()) / (admission_Data.max() - admission_Data.min())

normalized_admission_data_sampled = normalized_admission_data

split_index = int(0.8 * len(normalized_admission_data_sampled))

train_data = normalized_admission_data_sampled[:split_index]
test_data = normalized_admission_data_sampled[split_index:]



train_data_input = train_data.iloc[:,:-1]
train_data_expected_output = train_data.iloc[:,-1]

test_data_input = test_data.iloc[:,:-1]
test_data_expected_output = test_data.iloc[:,-1]


train_data_input = train_data_input.to_numpy()
train_data_expected_output = train_data_expected_output.to_numpy().reshape(-1, 1)



test_data_input = test_data_input.to_numpy()
test_data_expected_output = test_data_expected_output.to_numpy().reshape(-1,1)



model = NeuralNetwork(7)
model.add_layer(8,Activations.RELU)
model.add_layer(4,Activations.RELU)
model.add_layer(1,Activations.LINEAR)

model.train(train_data_input,train_data_expected_output,epochs=5)

model.test(test_data_input,test_data_expected_output)







