import numpy as np
from data import LabeledData
from network import Network


class Evaluator:
    def __init__(self, network: Network) -> None:
        self.network = network
    
    @staticmethod
    def indexOfMax(v: np.array) -> int:
        return max(enumerate(v), key=lambda x: x[1])[0]

    def evaluate(self, data: list[LabeledData]):
        correct = 0
        for d in data:
            result = self.network.apply(d.value)
            correct += self.indexOfMax(result) == self.indexOfMax(d.label)

        return correct / len(data)
