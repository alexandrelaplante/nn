import numpy as np

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


class ReluLayer:
    @staticmethod
    def f(x):
        return ReLU(x)
    
    @staticmethod
    def f_prime(x):
        return dReLU(x)

    def __init__(self, w: np.array, b: np.array) -> None:
        self.w = w
        self.b = b
    
    def apply(self, x: np.array) -> np.array:
        return self.f(self.z(x))
    
    def z(self, x: np.array) -> np.array:
        return self.w @ x + self.b

    def __repr__(self) -> str:
        return f'<ReluLayer: w={self.w.shape}, b={self.b.shape}>'
