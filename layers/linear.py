import numpy as np


class LinearLayer:
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def f_prime(x):
        return 1

    def __init__(self, w: np.array, b: np.array) -> None:
        self.w = w
        self.b = b

    def apply(self, x: np.array) -> np.array:
        return self.f(self.z(x))

    def z(self, x: np.array) -> np.array:
        return self.w @ x + self.b

    def __repr__(self) -> str:
        return f"<LinearLayer: w={self.w.shape}, b={self.b.shape}>"
