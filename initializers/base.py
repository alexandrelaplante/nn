import np
from abc import ABC, abstractclassmethod


class LayerInitializer(ABC):
    @abstractclassmethod
    def w(sizes: list[float]) -> np.ndarray: ...

    @abstractclassmethod
    def b(sizes: list[float]) -> np.ndarray: ...
