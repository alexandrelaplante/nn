import numpy as np


class InputLayer:
    def __init__(self) -> None:
        self.w = np.random.normal(size=[0, 0])
        self.b = np.random.normal(size=[0])

    def apply(self, x: np.array) -> np.array:
        return x


class LayerInitializer:
    @classmethod
    def w(cls, size):
        # return np.random.normal(size=size, scale=np.sqrt(size[1]/2))
        return np.random.normal(size=size)

    @classmethod
    def b(cls, size):
        return np.random.normal(size=size)


class Network:
    def __init__(self, sizes: list[int], layer_classes: list[type]) -> None:
        num_layers = len(sizes) - 1
        w_sizes = list(zip(sizes[1:], sizes[:-1]))
        b_sizes = list(zip(sizes[1:], [1] * num_layers))

        self.layers = [InputLayer()] + [
            layer_classes[i](
                w=LayerInitializer.w(w_sizes[i]),
                b=LayerInitializer.b(b_sizes[i]),
            )
            for i in range(num_layers)
        ]

    def apply(self, x: np.array) -> np.array:
        for layer in self.layers:
            x = layer.apply(x)
        return x


if __name__ == "__main__":
    from sigmoid import SigmoidLayer
    from relu import ReluLayer

    x = np.array([[0.5, 0.5, 0.5, 0.5]]).T
    n = Network(sizes=[4, 15, 10], layer_classes=[ReluLayer, SigmoidLayer])
    print(n.layers[0])
    print(n.layers[1])
    o = n.apply(x)
    print(o)
