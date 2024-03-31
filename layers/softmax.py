import numpy as np


# def softmax(z: np.ndarray) -> np.ndarray:
#     return np.exp(z) / np.sum(np.exp(z))


def softmax(z: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()


class SoftmaxLayer:
    @staticmethod
    def f(x):
        return softmax(x)

    @staticmethod
    def f_prime(x):
        # raise NotImplementedError
        return 1

    def __init__(self, w: np.array, b: np.array) -> None:
        self.w = w
        self.b = b

    def apply(self, x: np.array) -> np.array:
        return self.f(self.z(x))

    def z(self, x: np.array) -> np.array:
        return self.w @ x + self.b

    def __repr__(self) -> str:
        return f"<SoftmaxLayer: w={self.w.shape}, b={self.b.shape}>"
