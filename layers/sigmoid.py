import numpy as np


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


def sigmoid(z):
    # https://stackoverflow.com/a/62860170
    # to avoid overflows
    return np.piecewise(
        z,
        [z > 0],
        [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
    )


def sigmoid_prime(z):
    sz = sigmoid(z)
    return sz * (1 - sz)


class Sigmoid:
    def __init__(self, w, b, name=None):
        self.w = np.array(w)
        self.b = b
        self.name = name

    def apply(self, x):
        x = np.array(x)
        w = self.w
        b = self.b

        z = np.dot(w, x) + b
        return sigmoid(z)

    def __repr__(self):
        return self.name


class SigmoidLayer:
    @staticmethod
    def f(x):
        return sigmoid(x)

    @staticmethod
    def f_prime(x):
        return sigmoid_prime(x)

    def __init__(self, w: np.array, b: np.array) -> None:
        self.w = w
        self.b = b

    def apply(self, x: np.array) -> np.array:
        return self.f(self.z(x))

    def z(self, x: np.array) -> np.array:
        return self.w @ x + self.b

    def __repr__(self) -> str:
        return f"<SigmoidLayer: w={self.w.shape}, b={self.b.shape}>"
