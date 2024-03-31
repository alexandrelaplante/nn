import np
from abc import ABC, abstractstaticmethod


class Layer(ABC):
    @abstractstaticmethod
    def f(x): ...

    @abstractstaticmethod
    def f_prime(x): ...

    def __init__(self, w: np.array, b: np.array) -> None:
        self.w = w
        self.b = b

    def apply(self, x: np.array) -> np.array:
        return self.f(self.z(x))

    def z(self, x: np.array) -> np.array:
        return self.w @ x + self.b

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: w={self.w.shape}, b={self.b.shape}>"
