from layers.base import Layer
import np


class Relu(Layer):
    @staticmethod
    def f(x):
        return x * (x > 0)

    @staticmethod
    def f_prime(x):
        return 1.0 * (x > 0)
