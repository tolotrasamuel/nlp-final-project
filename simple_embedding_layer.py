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
        input_error = np.dot(output_error, self.weights.T)

        # Update parameters
        self.weights -= learning_rate * weights_error

        return input_error

    def forward_propagation(self, input_data):
        assert len(input_data) == 1

        # input_data should be an array of indices
        self.input = input_data
        self.output = self.weights[input_data]
        return self.output

    def get(self, index):
        return self.weights[index]

    def __str__(self):
        return f"SimpleEmbeddingLayer(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"

    def __repr__(self):
        return self.__str__()
    def clip_grads(self, max_norm):
        # normalize and clip weights
        norm = np.linalg.norm(self.weights)
        if norm > max_norm:
            self.weights *= max_norm / norm

        # normalize and clip biases
        norm = np.linalg.norm(self.bias)
        if norm > max_norm:
            self.bias *= max_norm / norm




class GlobalAveragePooling1D(Layer):
    def __init__(self):
        pass

    def backward_propagation(self, output_error, learning_rate):
        return output_error

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.mean(input_data, axis=1)
        return self.output

    def clip_grads(self, max_norm):
        pass
