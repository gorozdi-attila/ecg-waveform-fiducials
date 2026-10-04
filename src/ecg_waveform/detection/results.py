from dataclasses import dataclass
from typing import Any

from ecg_waveform.core import ECGAnnotation


@dataclass(frozen=True, slots=True)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "n_points": len(self.annotation.sample),
            "runtime_ms": self.runtime_s * 1000,
            "params": self.params,
        }
