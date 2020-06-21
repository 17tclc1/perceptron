import numpy as np

# Basic Rosenblatt Perceptron implementation
class Perceptron:
  # Constructor
  def __init__(self, iterations, learning_rate):
    self.iterations = iterations
    self.learning_rate = learning_rate

  # Train perceptron
  def train(self, data, RealOutput):
    # Initialize weights vector with ones
    self.weight = [1.0,1.0,1.0]
    # Perform the iterations
    for i in range(self.iterations):
      for sample, realoutput in zip(data, RealOutput):
        # Generate prediction and compare with real output
        predictedOutput = self.predict(sample)
        sample = np.insert(sample,0,1.)

        # Compute weight update via Perceptron Learning Rule
        # if realoutput = predictedOutput -> Do nothing to the weight
        # if realoutput > predictedOutput -> Add to weight
        # if realoutput < predictedOutput -> Subtract to weight

        if predictedOutput != realoutput:
          self.weight +=  self.learning_rate * (realoutput - predictedOutput) * sample  # Update weight
        else: # Do no thing if equal
          continue
    return self.weight

  # Generate prediction
  def predict(self, sample):
    # y = (w1.x1 + w2.x2) + w0
    predictedOutput = np.dot(sample, self.weight[1:]) + self.weight[0]
    return np.where(predictedOutput > 0, 1, 0)