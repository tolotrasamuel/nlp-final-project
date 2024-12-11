import os
import pickle

class Network:
    def __init__(self, verbose=False):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.verbose = verbose

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = self.predict_single(input_data[i])
            result.append(output)

        return result

    def predict_single(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = self.predict_single(x_train[j])

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def backprop(self, error: float, learning_rate: float = 0.01):
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate=learning_rate)

    def clip_grads(self, max_norm: float) -> None:
        for layer in self.layers:
            layer.clip_grads(max_norm)

    def save(self, filename):
        dir_name = os.path.dirname(filename)

        # Create the directory if it does not already exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(filename, 'wb') as f:
            print(f"Saving model to {filename}...")
            pickle.dump(self, f)