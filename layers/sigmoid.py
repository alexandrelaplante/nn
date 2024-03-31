from layers.base import Layer
import np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# def sigmoid(z):
#     # https://stackoverflow.com/a/62860170
#     # to avoid overflows
#     return np.piecewise(
#         z,
#         [z > 0],
#         [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
#     )


def sigmoid_prime(z):
    sz = sigmoid(z)
    return sz * (1 - sz)


class Sigmoid(Layer):
    @staticmethod
    def f(x):
        return sigmoid(x)

    @staticmethod
    def f_prime(x):
        return sigmoid_prime(x)
