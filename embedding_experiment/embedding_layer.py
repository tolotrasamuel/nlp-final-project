import numpy as np

from layer import Layer


class SimpleEmbeddingLayer(Layer):
    def init(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = np.random.rand(num_embeddings, embedding_dim) - 0.5

    def backward(self, output_error, learning_rate):
        # Derivative of the error with respect to the weights
        weights_error = np.zeros_like(self.weights)
        np.add.at(weights_error, self.input, output_error)

        # Derivative of the error with respect to the input
        input_error = np.dot(output_error, self.weights)

        # Update parameters
        self.weights -= learning_rate * weights_error

        return input_error

    def forward(self, input_data):
        # input_data should be an array of indices
        self.input = input_data
        self.output = self.weights[input_data]
        return self.output