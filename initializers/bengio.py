from initializers.base import LayerInitializer
import np


class Bengio(LayerInitializer):
    @classmethod
    def w(cls, size) -> np.ndarray:
        return np.random.normal(
            size=size, scale=1 / (4 * np.sqrt(6 * (size[0] + size[1])))
        )

    @classmethod
    def b(cls, size) -> np.ndarray:
        return np.zeros(shape=size)
