import numpy as np
import random
from data import LabeledData
from functools import cache
from tqdm import tqdm

from evaluate import Evaluator
from mnist import MNISTLoader
from network import Network


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class StochasticGradientDescent:
    def __init__(self, network: Network):
        self.network = network

    def feedforward(self, data: LabeledData) -> tuple[list[np.array], list[np.array]]:
        sz = [None] # need a dummy layer to represent input
        activations = [data.value]
        for layer in self.network.layers[1:]:
            z = layer.z(activations[-1])
            sz.append(layer.f_prime(z))
            activations.append(layer.f(z))
        
        return sz, activations
    
    def _partials(self) -> tuple[callable, callable]:
        @cache
        def feedforward(data: LabeledData):
            return self.feedforward(data)
        
        @cache
        def C_a(data: LabeledData, l: int) -> np.array:
            sz, activations = feedforward(data)

            if l == len(self.network.layers) - 1:
                return 2 * (activations[-1] - data.label)

            w = self.network.layers[l+1].w

            arr2 = sz[l+1]
            arr3 = C_a(data, l + 1)

            return w.T @ (arr2 * arr3)

        @cache
        def C_w(data: LabeledData, l: int) -> np.array:
            sz, activations = feedforward(data)

            k_piece = activations[l-1]
            j_piece = sz[l] * C_a(data, l)
            
            return np.outer(j_piece, k_piece)
        
        @cache
        def C_b(data: LabeledData, l: int) -> np.array:
            sz, activations = feedforward(data)
            return sz[l] * C_a(data, l)

        return C_w, C_b

    def _update(self, batch: list[LabeledData], learning_rate: float) -> None:
        layers = self.network.layers
        batched_delta_w = [np.zeros(shape=layer.w.shape) for layer in layers]
        batched_delta_b = [np.zeros(shape=layer.b.shape) for layer in layers]

        # C_w is partial derivative of cost(data) w.r.t. w_jkl
        # C_b is partial derivative of cost(data) w.r.t. b_jl
        C_w, C_b = self._partials()
        
        # Skip the input layer
        for l in range(1, len(layers)):
            step_size = learning_rate / len(batch)
            batched_delta_w[l] -= step_size * sum(C_w(data, l) for data in batch)
            batched_delta_b[l] -= step_size * sum(C_b(data, l) for data in batch)
        
        # Update network
        for l, layer in enumerate(layers):
            layer.w += batched_delta_w[l]
            layer.b += batched_delta_b[l]

    def train(
        self,
        training_data: list[LabeledData],
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        total = epochs * len(training_data) / batch_size
        with tqdm(total=total) as t:
            for _ in range(epochs):
                random.shuffle(training_data)

                # import cProfile
                # with cProfile.Profile() as pr:
                for batch in chunks(training_data, batch_size):
                    self._update(batch, learning_rate)
                    t.update()
                    # pr.print_stats('cumtime')
                    # exit()


if __name__ == '__main__':
    from sigmoid import SigmoidLayer
    from relu import ReluLayer
    data = MNISTLoader.load()
    n = Network(sizes=[784, 10, 10], layer_classes=[SigmoidLayer, SigmoidLayer])
    sgd = StochasticGradientDescent(n)

    evalator = Evaluator(n)
    accuracy = evalator.evaluate(data.test)
    print('pre-training accuracy', accuracy)

    sgd.train(
        training_data=data.training,
        epochs=3,
        batch_size=100,
        learning_rate=3.,
    )

    accuracy = evalator.evaluate(data.test)
    print('post-training accuracy', accuracy)
