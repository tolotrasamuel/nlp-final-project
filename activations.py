import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.max([x, 0])

def relu_prime(x):
    return 1 if x > 0 else 0

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)