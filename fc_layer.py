from layer import Layer
import numpy as np


class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def __repr__(self):
        return f"FCLayer(input_size={self.input_size}, output_size={self.output_size})"


    def clip_grads(self, max_norm):
        # normalize and clip weights
        norm = np.linalg.norm(self.weights)
        if norm > max_norm:
            self.weights *= max_norm / norm

        # normalize and clip biases
        norm = np.linalg.norm(self.bias)
        if norm > max_norm:
            self.bias *= max_norm / norm
