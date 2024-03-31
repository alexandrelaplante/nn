import numpy as np
import random
from cost import CostFunction
from data import LabeledData
from tqdm import tqdm

from network import Network
from stopping import StoppingCondition


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class StochasticGradientDescent:
    def __init__(self, network: Network, cost: type[CostFunction]):
        self.network = network
        self.cost = cost(network=network, feedforward=self._feedforward)

    def _feedforward(self, data: LabeledData) -> tuple[list[np.array], list[np.array]]:
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

    def _update(
        self,
        batch: list[LabeledData],
        learning_rate: float,
        reg_param: float,
    ) -> None:
        """All examples in the batch are done simultaneously"""
        data = self._matrixify(batch)
        step_size = learning_rate / len(batch)

        # C_w is partial derivative of cost(data) w.r.t. w_jkl
        # C_b is partial derivative of cost(data) w.r.t. b_jl
        C_w, C_b = self.cost.partials()

        # Skip the input layer
        for l, layer in enumerate(self.network.layers[1:], start=1):
            layer.w -= step_size * (reg_param * layer.w + C_w(data, l))
            layer.b -= step_size * np.sum(C_b(data, l), axis=1).reshape(layer.b.shape)

    def train(
        self,
        training_data: list[LabeledData],
        batch_size: int,
        learning_rate: float,
        stopping: StoppingCondition,
        reg_param: float = 0.0,
    ):
        estimate = stopping.estimate(0)
        total = None
        if estimate:
            total = estimate * len(training_data) / batch_size
        epoch_num = 1
        postfix = {"epochs": epoch_num, "accuracy": "0%"}
        with tqdm(total=total, unit=" samples", postfix=postfix) as t:
            while not stopping.should_stop(epoch_num):
                postfix = {"epochs": epoch_num}
                accuracy = stopping.accuracy(epoch_num)
                epoch_num += 1
                if accuracy is not None:
                    postfix["accuracy"] = f"{100*stopping.accuracy(epoch_num)}%"
                t.set_postfix(postfix)

                random.shuffle(training_data)
                for batch in chunks(training_data, batch_size):
                    self._update(
                        batch,
                        learning_rate=learning_rate,
                        # rewrite our lambda as m * lambda / n
                        # because our code's weight decay is (1 - eta * lambda / m)
                        # this results in weight decay of (1 - eta * lambda / n)
                        # as expected
                        reg_param=len(batch) * reg_param / len(training_data),
                        # reg_param=reg_param,
                    )
                    t.update()
