import numpy as np
from layer import Layer

class SimpleEmbeddingLayer(Layer):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = np.random.rand(num_embeddings, embedding_dim) - 0.5

    def backward_propagation(self, output_error, learning_rate):
        # Derivative of the error with respect to the weights
        weights_error = np.zeros_like(self.weights)
        np.add.at(weights_error, self.input, output_error)

        # Derivative of the error with respect to the input
        input_error = np.dot(output_error, self.weights)

        # Update parameters
        self.weights -= learning_rate * weights_error

        return input_error

    def forward_propagation(self, input_data):
        # input_data should be an array of indices
        self.input = input_data
        self.output = self.weights[input_data]
        return self.output

    def get(self, index):
        return self.weights[index]

    def clip_grads(self, max_norm):
        # normalize and clip weights
        norm = np.linalg.norm(self.weights)
        if norm > max_norm:
            self.weights *= max_norm / norm

        # normalize and clip biases
        norm = np.linalg.norm(self.bias)
        if norm > max_norm:
            self.bias *= max_norm / norm
