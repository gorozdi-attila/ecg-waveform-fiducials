from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from time import perf_counter

import numpy as np

from ecg_waveform.core import (
    Configurable,
    ECGAnnotation,
    ECGSignal,
    Fiducials,
)

from .results import DetectionResult


def restrict_to_points(
    owner: str,
    supported: frozenset[Fiducials],
    annotation: ECGAnnotation,
    points: Iterable[Fiducials] | None,
) -> ECGAnnotation:
    requested = frozenset(points) if points is not None else supported
    unsupported = requested - supported
    if unsupported:
        raise ValueError(
            f"{owner!r} does not support point types: "
            f"{sorted(p.value for p in unsupported)}"
        )
    return annotation.filter(requested)


def prepare_r_peaks(
    signal: ECGSignal, r_peaks: np.ndarray | ECGAnnotation
) -> np.ndarray:
    if isinstance(r_peaks, ECGAnnotation):
        r_peaks = r_peaks.filter(Fiducials.R_PEAK).sample
    r_peaks = np.asarray(r_peaks)

    if r_peaks.ndim != 1:
        raise ValueError("r_peaks must be one-dimensional.")
    if len(r_peaks) == 0:
        return r_peaks.astype(np.int64)
    if not np.all(np.isfinite(r_peaks)):
        raise ValueError("r_peaks contains non-finite values.")
    if np.any(np.diff(r_peaks) <= 0):
        raise ValueError("r_peaks must be strictly increasing.")
    if np.any(r_peaks < 0) or np.any(r_peaks >= signal.n_samples):
        raise ValueError("r_peaks contains out-of-range samples.")
    return r_peaks


def _without_reference(signal: ECGSignal) -> ECGSignal:
    return signal if signal.annotation is None else signal.with_annotation(None)


def _timed_result(
    owner: "BasePQRSTDetector | BasePQRSTDelineator",
    signal: ECGSignal,
    compute: Callable[[ECGSignal], ECGAnnotation],
) -> DetectionResult:
    signal = _without_reference(signal)
    start = perf_counter()
    annotation = compute(signal)
    runtime_s = perf_counter() - start

    return DetectionResult(
        detector_name=owner.name,
        annotation=annotation,
        runtime_s=runtime_s,
        params=owner.get_params(),
    )


class _BaseDetector(Configurable, ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def supported_points(self) -> frozenset[Fiducials]:
        ...

    def _validate_points(
        self,
        points: Iterable[Fiducials] | None,
    ) -> frozenset[Fiducials]:
        requested = (
            frozenset(points)
            if points is not None
            else self.supported_points
        )
        unsupported = requested - self.supported_points

        if unsupported:
            names = sorted(p.value for p in unsupported)
            raise ValueError(
                f"{self.label!r} does not support point types: {names}"
            )

        return requested

    def _run(
        self,
        signal: ECGSignal,
        compute: Callable[[ECGSignal], ECGAnnotation],
    ) -> DetectionResult:
        return _timed_result(self, signal, compute)


class BasePQRSTDetector(_BaseDetector, ABC):
    __slots__ = ()

    @abstractmethod
    def detect(self, signal: ECGSignal) -> ECGAnnotation:
        ...

    def detect_points(
        self,
        signal: ECGSignal,
        points: Iterable[Fiducials] | None = None,
    ) -> ECGAnnotation:
        requested = self._validate_points(points)
        return self.detect(signal).filter(requested)

    def run(self, signal: ECGSignal) -> DetectionResult:
        return self._run(signal, self.detect)


class BasePQRSTDelineator(_BaseDetector, ABC):
    __slots__ = ()

    @abstractmethod
    def delineate(
        self,
        signal: ECGSignal,
        r_peaks: np.ndarray | ECGAnnotation,
    ) -> ECGAnnotation:
        ...

    def delineate_points(
        self,
        signal: ECGSignal,
        r_peaks: np.ndarray | ECGAnnotation,
        points: Iterable[Fiducials] | None = None,
    ) -> ECGAnnotation:
        requested = self._validate_points(points)
        peaks = prepare_r_peaks(signal, r_peaks)
        return self.delineate(signal, peaks).filter(requested)

    def run(
        self,
        signal: ECGSignal,
        r_peaks: np.ndarray | ECGAnnotation,
    ) -> DetectionResult:
        return self._run(
            signal,
            lambda s: self.delineate(s, prepare_r_peaks(s, r_peaks)),
        )