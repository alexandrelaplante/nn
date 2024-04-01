from functools import cache
import np
from data import DataSet
from evaluate import Evaluator
from network import Network
from abc import ABC, abstractmethod


class StoppingCondition(ABC):
    def __init__(self, net: Network, data: DataSet) -> None:
        self.net = net
        self.data = data
        self.eval = Evaluator(net)

    @abstractmethod
    def should_stop(self, epoch_num: int) -> bool: ...

    @abstractmethod
    def estimate(self, epoch_num: int = 0) -> int | None: ...

    @cache
    def accuracy(self, epoch_num: int) -> float | None:
        return self.eval.evaluate(self.data.validation)


class AverageImprovement(StoppingCondition):
    def __init__(self, threshold=0.001, lookback=20, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.lookback = lookback
        self.improvements = []
        self.last_accuracy = 0.0

    def should_stop(self, epoch_num: int) -> bool:
        accuracy = self.accuracy(epoch_num)
        self.improvements.append(accuracy - self.last_accuracy)
        self.last_accuracy = accuracy

        if len(self.improvements) <= self.lookback:
            return False

        return np.mean(self.improvements[-self.lookback :]) < self.threshold

    def estimate(self, epoch_num: int) -> int | None:
        return None


class LastImprovement(StoppingCondition):
    def __init__(self, lookback=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lookback = lookback
        self.best_accuracy = 0.0
        self.epochs_since_best = 0

    def should_stop(self, epoch_num: int) -> bool:
        accuracy = self.accuracy(epoch_num)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.epochs_since_best = 0
        self.epochs_since_best += 1

        return self.epochs_since_best > self.lookback

    def estimate(self, epoch_num: int) -> int | None:
        return None


class Epochs(StoppingCondition):
    def __init__(self, epochs: int = 10, show_accuracy=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self.show_accuracy = show_accuracy  # set to False for performance

    def should_stop(self, epoch_num: int) -> bool:
        return epoch_num > self.epochs

    def estimate(self, epoch_num: int = 0) -> int | None:
        return self.epochs - epoch_num

    def accuracy(self, *args, **kwargs) -> float | None:
        if not self.show_accuracy:
            return None
        return super().accuracy(*args, **kwargs)
