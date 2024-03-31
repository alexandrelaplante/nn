from initializers.base import LayerInitializer
import np


class Normal(LayerInitializer):
    @classmethod
    def w(cls, size) -> np.ndarray:
        return np.random.normal(size=size)

    @classmethod
    def b(cls, size) -> np.ndarray:
        return np.random.normal(size=size)
