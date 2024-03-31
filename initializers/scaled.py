from initializers.base import LayerInitializer
import np


class ScaledNormal(LayerInitializer):
    @classmethod
    def w(cls, size) -> np.ndarray:
        return np.random.normal(size=size, scale=1 / np.sqrt(size[1]))

    @classmethod
    def b(cls, size) -> np.ndarray:
        return np.random.normal(size=size)
