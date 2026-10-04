from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from time import perf_counter
from typing import Any

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials

from ..results import DetectionResult


class BaseDelineator(ABC):
    __slots__ = ()

    @property
    def supported_points(self) -> frozenset[Fiducials]:
        return frozenset(Fiducials)

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
    def delineate(self, signal: ECGSignal, r_peaks: ECGAnnotation) -> list[ECGAnnotation]: ...

    def detect_points(
        self,
        signal: ECGSignal,
        points: Iterable[Fiducials] | None = None,
    ) -> ECGAnnotation:
        requested = frozenset(points) if points is not None else self.supported_points
    
        unsupported = requested - self.supported_points
        if unsupported:
            raise ValueError(
                f"{self.name!r} does not support point types: "
                f"{sorted(p.value for p in unsupported)}"
            )
    
        annotation = self.detect(signal)
        return annotation.filter([p.value for p in requested])
    
    def detect_point(self, signal: ECGSignal, point: Fiducials) -> ECGAnnotation:
        return self.detect_points(signal, points=[point])

    def run(self, signal: ECGSignal) -> DetectionResult:
        start = perf_counter()
        annotation = self.delineate(signal)
        end = perf_counter()

        return DetectionResult(
            detector_name=self.name,
            annotation=annotation,
            runtime_s=end - start,
            params=self.get_params(),
        )

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.name}({params})"
