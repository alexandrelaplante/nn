import numpy as np
from data import LabeledData
from functools import cache
from abc import ABC, abstractmethod


class CostFunction(ABC):
    def __init__(self, network, feedforward) -> None:
        self.network = network
        self.feedforward = feedforward

    @abstractmethod
    def partials(self) -> tuple[callable, callable]: ...


class Quadratic(CostFunction):
    """Used with Sigmoid output layer."""

    def partials(self) -> tuple[callable, callable]:
        @cache
        def feedforward(data: LabeledData):
            return self.feedforward(data)

        @cache
        def C_a(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)

            if l == len(self.network.layers) - 1:
                # TODO: Why is sz[-1] in here? I thought if
                # C = ½*(a - y)**2
                # δC/δa = a - y
                # which would just give us what is currently implemented
                # as CrossEntropy
                return sz[-1] * (activations[-1] - data.label)

            w = self.network.layers[l + 1].w

            arr2 = sz[l + 1]
            arr3 = C_a(data, l + 1)

            return w.T @ (arr2 * arr3)

        @cache
        def delta(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)
            return sz[l] * C_a(data, l)

        @cache
        def C_w(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)
            return delta(data, l) @ activations[l - 1].T

        @cache
        def C_b(data: LabeledData, l: int) -> np.ndarray:
            return delta(data, l)

        return C_w, C_b


class CrossEntropy(CostFunction):
    """Used for sigmoid output layer."""

    def partials(self) -> tuple[callable, callable]:
        @cache
        def feedforward(data: LabeledData):
            return self.feedforward(data)

        @cache
        def C_a(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)

            if l == len(self.network.layers) - 1:
                return activations[-1] - data.label

            w = self.network.layers[l + 1].w

            arr2 = sz[l + 1]
            arr3 = C_a(data, l + 1)

            return w.T @ (arr2 * arr3)

        @cache
        def delta(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)
            return sz[l] * C_a(data, l)

        @cache
        def C_w(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)
            return delta(data, l) @ activations[l - 1].T

        @cache
        def C_b(data: LabeledData, l: int) -> np.ndarray:
            return delta(data, l)

        return C_w, C_b


class LogLikelihood(CrossEntropy):
    "Used with softmax output layer"
