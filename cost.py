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
                return activations[-1] - data.label

            w = self.network.layers[l + 1].w

            arr2 = sz[l + 1]
            arr3 = C_a(data, l + 1)

            return w.T @ (arr2 * arr3)

        @cache
        def C_w(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)

            k_piece = activations[l - 1]
            j_piece = sz[l] * C_a(data, l)

            return j_piece @ k_piece.T

        @cache
        def C_b(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)
            return sz[l] * C_a(data, l)

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

            return w.T @ C_a(data, l + 1)

        @cache
        def C_w(data: LabeledData, l: int) -> np.ndarray:
            sz, activations = feedforward(data)

            k_piece = activations[l - 1]
            j_piece = C_a(data, l)

            return j_piece @ k_piece.T

        @cache
        def C_b(data: LabeledData, l: int) -> np.ndarray:
            return C_a(data, l)

        return C_w, C_b


class LogLikelihood(CrossEntropy):
    "Used with softmax output layer"
