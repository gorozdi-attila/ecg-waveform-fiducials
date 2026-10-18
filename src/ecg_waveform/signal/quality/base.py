from abc import ABC, abstractmethod
from enum import Enum, IntEnum

from ecg_waveform.core import Configurable, ECGSignal


class Direction(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_RANGE = "target_range"


class QualityLevel(IntEnum):
    BAD = 0
    ACCEPTABLE = 1
    GOOD = 2


class SignalQualityMetric(Configurable, ABC):
    __slots__ = ()

    @abstractmethod
    def compute(self, signal: ECGSignal) -> float: ...
