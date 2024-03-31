from initializers import LayerInitializer
from layers import Layer
import np


class Network:
    def __init__(
        self, layers: list[tuple[float, type[Layer], type[LayerInitializer]]]
    ) -> None:
        self.layers = []
        for i, layer in enumerate(layers):
            size, klass, init = layer
            last_size = layers[i - 1][0]
            self.layers.append(
                klass(
                    w=init.w((size, last_size)),
                    b=init.b((size, 1)),
                )
            )

    def apply(self, x: np.array) -> np.array:
        for layer in self.layers[1:]:  # Skip input layer
            x = layer.apply(x)
        return x
