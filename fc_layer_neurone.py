from layer import Layer
import numpy as np

from neurone import Neuron

np.random.seed(0)

# inherit from base class Layer
class FCLayerDetailed(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __repr__(self):
        return f"FCLayer(input_size={len(self.neurones)}, output_size={self.output_size})"

    def __init__(self, input_size, output_size):
        self.output_size = output_size
        self.neurones = [Neuron(output_size) for _ in range(input_size)]
        self.bias = np.random.rand(output_size) - 0.5

        self.output = [None for _ in range(output_size)]
    # returns output for a given input

    def forward_propagation(self, input_data):

        assert len(input_data) == 1
        for input_index in range(len(self.neurones)):
            self.neurones[input_index].input = input_data[0][input_index]

        output = []
        for output_index in range(self.output_size):
            output_neurone_val = 0
            for input_index in range(len(self.neurones)):
                neurone = self.neurones[input_index]
                output_neurone_val += neurone.input * neurone.weights[output_index]
            output_neurone_val += self.bias[output_index]
            output.append(output_neurone_val)
        assert len(output) == self.output_size
        self.output = np.array(output)
        return np.array([self.output])

    # for a given output_error=dE / dY derivative of error with respect to output
    # which is equal to dE / dB , the derivative of the error with respect to the bias
    # computes dE/dW, dE/dB.
    # Returns input_error=dE/dX. The derivative of the error with respect to the input
    # backward_output_error is the input from back ward
    def backward_propagation(self, backward_output_error_param, learning_rate):


        # validate
        # input_error = np.dot(backward_output_error_param, self.weights.T)
        # derivative of the error with respect to weights
        # input_val = np.array([np.array([neurone.input for neurone in self.neurones]) ])
        # weights = np.array([[x for x in neurone.weights] for neurone in self.neurones])
        # weights_error = np.dot(input_val.T, backward_output_error_param)
        # bias = np.array([[x for x in self.bias]])
        # # dBias = output_error
        #
        # # update parameters
        # weights -= learning_rate * weights_error
        # bias -= learning_rate * backward_output_error_param

        assert len(backward_output_error_param) == 1
        backward_output_error = backward_output_error_param[0]

        # input_error = np.dot(output_error, self.weights.T)
        # # derivative of the error with respect to weights
        # weights_error = np.dot(self.input.T, output_error)
        # # dBias = output_error
        backward_input_error = []
        for i in range(len(self.neurones)):
            input_neurone_val = 0
            neurone = self.neurones[i]
            for o in range(self.output_size):
                input_neurone_val += neurone.weights[o] * backward_output_error[o]
            backward_input_error.append(input_neurone_val)
        assert len(backward_input_error) == len(self.neurones)

        # weight adjustment
        for i in range(len(self.neurones)):
            neurone = self.neurones[i]
            for o in range(self.output_size):
               neurone.weights[o] -= learning_rate * backward_output_error[o] * neurone.input

        # bias adjustment
        for o in range(self.output_size):
            self.bias[o] -= learning_rate * backward_output_error[o]


        return np.array(backward_input_error)
