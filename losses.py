import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def crossentropyloss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


def cross_entropy_loss_function(y_true, y_pred):
    epsilon = 1e-10
    loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    return loss


def crossentropyloss_prime(y_true, y_pred):
    epsilon = 1e-10
    gradient = -y_true / (y_pred + epsilon)
    return gradient

