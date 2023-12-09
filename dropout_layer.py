from layer import Layer
import numpy as np


# # inherit from base class Layer
# class FCLayer(Layer):
#     # input_size = number of input neurons
#     # output_size = number of output neurons
#     def __repr__(self):
#         return f"FCLayer(input_size={self.input_size}, output_size={self.output_size})"

#     def __init__(self, input_size, output_size):
#         self.input_size = input_size
#         self.output_size = output_size
#         self.weights = np.random.rand(input_size, output_size) - 0.5
#         self.bias = np.random.rand(1, output_size) - 0.5

#     # returns output for a given input
#     def forward_propagation(self, input_data):
#         self.input = input_data
#         self.output = np.dot(self.input, self.weights) + self.bias
#         return self.output

#     # for a given output_error=dE / dY derivative of error with respect to output
#     # which is equal to dE / dB , the derivative of the error with respect to the bias
#     # computes dE/dW, dE/dB.
#     # Returns input_error=dE/dX. The derivative of the error with respect to the input
#     def backward_propagation(self, output_error, learning_rate):
#         input_error = np.dot(output_error, self.weights.T)
#         # derivative of the error with respect to weights
#         weights_error = np.dot(self.input.T, output_error)
#         # dBias = output_error

#         # update parameters
#         self.weights -= learning_rate * weights_error
#         self.bias -= learning_rate * output_error
#         return input_error

class DropoutLayer(Layer):
	def __init__(self, p):
		self.p = p
		self.mask = None
		self.scale = 1 / (1 - p)

	def __repr__(self):
		return f"DropoutLayer(p={self.p})"
	
	def forward_propagation(self, input_data):
		self.input = input_data
		self.mask = np.random.binomial(1, 1 - self.p, size=input_data.shape)
		return self.input * self.mask * self.scale
	
	def backward_propagation(self, output_error, learning_rate):
		return output_error * self.mask * self.scale

	def clip_grads(self, max_norm):
		pass

