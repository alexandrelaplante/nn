import np
import random
from cost import CostFunction
from data import LabeledData


from network import Network
from progress_bar import ProgressBar
from regularization import Regularization
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

    def _reset_velocities(self) -> None:
        self.w_velocities = [
            np.zeros(shape=layer.w.shape) for layer in self.network.layers
        ]
        self.b_velocities = [
            np.zeros(shape=layer.b.shape) for layer in self.network.layers
        ]

    def _update(
        self,
        batch: list[LabeledData],
        learning_rate: float,
        regularization: Regularization,
        momentum_coefficient: float,
    ) -> None:
        """All examples in the batch are done simultaneously"""
        data = self._matrixify(batch)
        step_size = learning_rate / len(batch)

        # C_w is partial derivative of cost(data) w.r.t. w_jkl
        # C_b is partial derivative of cost(data) w.r.t. b_jl
        C_w, C_b = self.cost.partials()

        # Skip the input layer
        for l, layer in enumerate(self.network.layers[1:], start=1):
            w_v, b_v = self.w_velocities[l], self.b_velocities[l]

            self.w_velocities[l] = momentum_coefficient * w_v + regularization.apply(
                learning_rate=learning_rate,
                m=len(batch),
                w=layer.w,
                delta=C_w(data, l),
            )
            self.b_velocities[l] = momentum_coefficient * b_v - step_size * np.sum(
                C_b(data, l), axis=1
            ).reshape(layer.b.shape)
            layer.w += self.w_velocities[l]
            layer.b += self.b_velocities[l]

            # layer.w += regularization.apply(
            #     learning_rate=learning_rate,
            #     m=len(batch),
            #     w=layer.w,
            #     delta=C_w(data, l),
            # )
            # layer.b -= step_size * np.sum(C_b(data, l), axis=1).reshape(layer.b.shape)

    def train(
        self,
        training_data: list[LabeledData],
        batch_size: int,
        learning_rate: float,
        stopping: StoppingCondition,
        regularization: Regularization,
        momentum_coefficient: float = 0.0,
    ) -> None:
        estimate = stopping.estimate(0)
        total = None
        if estimate:
            total = estimate * len(training_data)
        epoch_num = 1

        p = ProgressBar(total=total)
        self._reset_velocities()
        while not stopping.should_stop(epoch_num):
            accuracy = stopping.accuracy(epoch_num)
            p.set_postfix(epoch_num, accuracy)
            epoch_num += 1

            random.shuffle(training_data)
            for batch in chunks(training_data, batch_size):
                self._update(
                    batch,
                    learning_rate=learning_rate,
                    momentum_coefficient=momentum_coefficient,
                    regularization=regularization,
                )
                p.update(n=len(batch))
