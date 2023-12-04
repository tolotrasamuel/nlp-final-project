import numpy as np


class Neuron:
    def __init__(self, output_size):
        self.output_size = output_size
        self.weights = np.random.rand(output_size) - 0.5
        self.bias: list[int] = np.random.rand(1) - 0.5
        self.output: list[int] = None
        self.input: list[int] = None


    def forward_propagation(self, input):
        self.output = np.dot(input, self.weights) + self.bias

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        # derivative of the error with respect to weights
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

