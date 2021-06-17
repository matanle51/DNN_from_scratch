import numpy as np


def relu(x):
    """
    ReLU activation function := max(x, 0)
    :param x: input to relu
    :return: relu result max(x, 0)
    """
    return np.maximum(x, 0)


def tanh(x):
    """
    Tanh activation function := (e^x-e^-x)/(e^x+e^-x)
    :param x: input to tanh
    :return: tanh result (e^x-e^-x)/(e^x+e^-x)
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))