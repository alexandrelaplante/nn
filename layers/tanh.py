from layers.base import Layer
import np


class Tanh(Layer):
    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def f_prime(x):
        return 1.0 - np.tanh(x) ** 2
