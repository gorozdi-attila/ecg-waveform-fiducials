from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum, IntEnum
from typing import Any

from ecg_waveform.core import ECGSignal


class Direction(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_RANGE = "target_range"


class QualityLevel(IntEnum):
    BAD = 0
    ACCEPTABLE = 1
    GOOD = 2


class SignalQualityMetric(ABC):
    __slots__ = ()

    @property
    def name(self) -> str:
        return type(self).__name__

    def get_params(self) -> dict[str, Any]:
        if is_dataclass(self):
            return asdict(self)
        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_") and not callable(v)
        }

    @abstractmethod
    def compute(self, signal: ECGSignal) -> float: ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.name}({params})"
