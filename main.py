# Import libraries
import numpy as np
from PerceptionClass import Perceptron
import matplotlib.pyplot as plt

import pandas as pd

dataFromFile = pd.read_csv('data_perceptron.csv').values
data = dataFromFile[:, [0, 1]]
RealOutput = dataFromFile[:,2]

positive = data[RealOutput.reshape(-1) == 1, :]
negative = data[RealOutput.reshape(-1) == 0, :]

plt.scatter(positive[:, 0], positive[:, 1], c='r')
plt.scatter(negative[:, 0], negative[:, 1], c='b')

x1_min = np.min(negative[:,0])
x2_max = np.max(positive[:,0])


# 10000 iterations, 0.001 learning rate
perceptron = Perceptron(10000, 0.001)

weight = perceptron.train(data, RealOutput)

y1_min = (-weight[0] - weight[1]*x1_min)/ weight[2]
y2_max = (- weight[0] - weight[1]*x2_max) / weight[2]

x_seperator = np.array([x1_min,x2_max])
y_seperator = np.array([y1_min,y2_max])

plt.plot(x_seperator,y_seperator,label = 'Seperation')

plt.title('Perceptron')
plt.xlabel('Lương')
plt.ylabel('Thời gian làm việc')

# result = trained_model.predict([5,0.8])
# print(result)
plt.show()