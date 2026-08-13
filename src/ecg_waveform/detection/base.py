from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from time import perf_counter
from typing import Any

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials


@dataclass(frozen=True)
class DetectionResult:
    detector_name: str
    annotation: ECGAnnotation
    runtime_s: float
    params: dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"DetectionResult(detector={self.detector_name!r}, "
            f"n_points={len(self.annotation.sample)}, "
            f"runtime={self.runtime_s * 1000:.1f} ms)"
        )


class BaseDetector(ABC):
    __slots__ = ()

    @property
    def supported_points(self) -> frozenset[Fiducials]:
        return frozenset(Fiducials)

    @property
    def name(self) -> str:
        return type(self).__name__

    def get_params(self) -> dict[str, Any]:
        if is_dataclass(self):
            return {f.name: getattr(self, f.name) for f in fields(self)}

        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_") and not callable(v)
        }

    @abstractmethod
    def detect(self, signal: ECGSignal) -> ECGAnnotation: ...

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
        annotation = self.detect(signal)
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
