from itertools import count
from dataclasses import dataclass, field
import np


@dataclass(frozen=True)
class LabeledData:
    value: np.ndarray
    label: np.ndarray
    id: int = field(default_factory=count().__next__)

    def __hash__(self) -> int:
        return self.id


@dataclass(frozen=True)
class DataSet:
    training: list[LabeledData]
    validation: list[LabeledData]
    test: list[LabeledData]
