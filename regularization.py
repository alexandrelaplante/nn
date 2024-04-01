import np


class Regularization:
    pass


class L1Regularization(Regularization):
    def __init__(self, lmbda: float, n: int) -> None:
        self.lmbda = lmbda
        self.n = n

    def apply(
        self,
        learning_rate: float,
        m: float,
        w: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        return -learning_rate * (self.lmbda * np.sign(w) / self.n + delta / m)


class L2Regularization(Regularization):
    def __init__(self, lmbda: float, n: int) -> None:
        self.lmbda = lmbda
        self.n = n

    def apply(
        self,
        learning_rate: float,
        m: float,
        w: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        return -learning_rate * (self.lmbda * w / self.n + delta / m)
