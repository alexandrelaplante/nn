import numpy as np
import random
from cost import CostFunction, Quadratic
from data import LabeledData
from tqdm import tqdm

from evaluate import Evaluator
from mnist import MNISTLoader
from network import Network


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class StochasticGradientDescent:
    def __init__(self, network: Network, cost: CostFunction):
        self.network = network
        self.cost = cost(network=network, feedforward=self.feedforward)

    def feedforward(self, data: LabeledData) -> tuple[list[np.array], list[np.array]]:
        sz = [None]  # need a dummy layer to represent input
        activations = [data.value]
        for layer in self.network.layers[1:]:
            z = layer.z(activations[-1])
            sz.append(layer.f_prime(z))
            activations.append(layer.f(z))

        return sz, activations

    @staticmethod
    def _matrixify(batch: list[LabeledData]) -> LabeledData:
        data_matrix = np.column_stack(
            [data.value.reshape(data.value.shape[0]) for data in batch]
        )
        label_matrix = np.column_stack(
            [data.label.reshape(data.label.shape[0]) for data in batch]
        )
        return LabeledData(value=data_matrix, label=label_matrix)

    def _update(self, batch: list[LabeledData], learning_rate: float) -> None:
        """All examples in the batch are done simultaneously"""
        data = self._matrixify(batch)
        step_size = -learning_rate / len(batch)

        # C_w is partial derivative of cost(data) w.r.t. w_jkl
        # C_b is partial derivative of cost(data) w.r.t. b_jl
        C_w, C_b = self.cost.partials()

        # Skip the input layer
        for l, layer in enumerate(self.network.layers[1:], start=1):
            layer.w += step_size * C_w(data, l)
            layer.b += step_size * np.sum(C_b(data, l), axis=1).reshape(layer.b.shape)

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


if __name__ == "__main__":
    from sigmoid import SigmoidLayer
    from relu import ReluLayer

    data = MNISTLoader.load()
    n = Network(
        sizes=[784, 10, 10],
        layer_classes=[SigmoidLayer, SigmoidLayer, SigmoidLayer],
    )
    sgd = StochasticGradientDescent(n, cost=Quadratic)

    evalator = Evaluator(n)
    accuracy = evalator.evaluate(data.test)
    print("pre-training accuracy", accuracy)

    sgd.train(
        training_data=data.training,
        epochs=3,
        batch_size=100,
        learning_rate=10.0,
    )

    accuracy = evalator.evaluate(data.test)
    print("post-training accuracy", accuracy)
