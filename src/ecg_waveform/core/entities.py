from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from math import gcd

import numpy as np
from scipy.signal import resample_poly


class Fiducials(str, Enum):
    P_WAVE = "P"
    Q_WAVE = "Q"
    R_WAVE = "R"
    S_WAVE = "S"
    T_WAVE = "T"


@dataclass(frozen=True, eq=False, repr=False)
class ECGAnnotation:
    symbol: np.ndarray
    sample: np.ndarray

    def __post_init__(self):
        if self.symbol.ndim != 1 or self.sample.ndim != 1:
            raise ValueError("symbol and sample must be 1-dimensional arrays")

        if len(self.symbol) != len(self.sample):
            raise ValueError("symbol and sample must have the same length")

        if len(self.sample) > 0:
            if not np.issubdtype(self.sample.dtype, np.integer):
                raise TypeError("sample must be an integer array (sample indices)")

            if np.any(self.sample < 0):
                raise ValueError("sample indices must be non-negative")

            if np.any(np.diff(self.sample) < 0):
                raise ValueError("sample indices must be sorted in ascending order")

        self.symbol.flags.writeable = False
        self.sample.flags.writeable = False

    def __len__(self) -> int:
        return len(self.sample)

    def __repr__(self) -> str:
        return f"Annotation(n_points={len(self.sample)}, unique_symbols={sorted(set(self.symbol.tolist()))})"

    def filter(self, symbols: str | list[str]) -> "ECGAnnotation":
        symbols = [symbols] if isinstance(symbols, str) else symbols

        mask = np.isin(self.symbol, symbols)

        return ECGAnnotation(
            symbol=self.symbol[mask],
            sample=self.sample[mask],
        )

    def merge(self, other: "ECGAnnotation") -> "ECGAnnotation":
        sample = np.concatenate([self.sample, other.sample])
        symbol = np.concatenate([self.symbol, other.symbol])

        order = np.argsort(sample, kind="stable")

        return ECGAnnotation(
            symbol=symbol[order],
            sample=sample[order]
        )

    def segment(self, start: int, end: int) -> "ECGAnnotation":
        if start < 0 or end < start:
            raise ValueError("Invalid segment bounds")
        
        mask = (self.sample >= start) & (self.sample < end)
        return ECGAnnotation(
            symbol=self.symbol[mask],
            sample=self.sample[mask] - start,
        )


@dataclass(frozen=True, eq=False)
class ECGSignal:
    sample: np.ndarray
    sample_rate: int
    channel: int
    lead_name: str
    annotation: ECGAnnotation | None = None

    def __post_init__(self):
        if self.sample.ndim != 1:
            raise ValueError("sample must be a 1-dimensional array")

        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

        if self.channel < 0:
            raise ValueError(f"channel must be non-negative, got {self.channel}")

        if self.n_samples == 0:
            raise ValueError("signal must contain at least one sample")

        self.sample.flags.writeable = False

        if self.annotation is not None:
            if np.any(self.annotation.sample < 0):
                raise ValueError("Annotation contains negative sample indices")
            if np.any(self.annotation.sample >= self.n_samples):
                raise ValueError("Annotation outside signal")

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        ann_info = (
            f", annotation={self.annotation!r}" if self.annotation is not None else ""
        )
        return (
            f"ECGSignal(channel={self.channel}, lead_name={self.lead_name!r}, "
            f"n_samples={self.n_samples}, sample_rate={self.sample_rate}, "
            f"duration={self.duration:.2f}s{ann_info})"
        )

    @property
    def n_samples(self) -> int:
        return len(self.sample)

    @property
    def duration(self) -> float:
        return len(self.sample) / self.sample_rate

    @property
    def is_annotated(self) -> bool:
        return self.annotation is not None

    @cached_property
    def time(self) -> np.ndarray:
        t = np.arange(self.n_samples) / self.sample_rate
        t.flags.writeable = False
        return t

    def segment(self, start: int, end: int) -> "ECGSignal":
        if not (0 <= start <= end <= self.n_samples):
            raise ValueError("Invalid segment bounds")

        sample_segment = self.sample[start:end]
        
        annotation_segment = None

        if self.annotation is not None:
            annotation_segment = self.annotation.segment(start, end)

        return ECGSignal(
            sample=sample_segment,
            sample_rate=self.sample_rate,
            channel=self.channel,
            lead_name=self.lead_name,
            annotation=annotation_segment,
        )

    def resample(self, target_sample_rate: int) -> "ECGSignal":
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")

        if self.sample_rate == target_sample_rate:
            return self

        sr = int(self.sample_rate)
        tsr = int(target_sample_rate)

        g = gcd(sr, tsr)

        samples = resample_poly(
            self.sample,
            up=tsr // g,
            down=sr // g,
        )

        annotation = None
        if self.annotation is not None:
            new_indices = np.rint(
                self.annotation.sample * target_sample_rate / self.sample_rate
            ).astype(int)

            new_indices = np.clip(new_indices, 0, len(samples) - 1)

            annotation = ECGAnnotation(
                symbol=self.annotation.symbol,
                sample=new_indices,
            )

        return ECGSignal(
            sample=samples,
            sample_rate=target_sample_rate,
            channel=self.channel,
            lead_name=self.lead_name,
            annotation=annotation,
        )

    def with_sample(self, sample: np.ndarray) -> "ECGSignal":
        return ECGSignal(
            sample=sample,
            sample_rate=self.sample_rate,
            channel=self.channel,
            lead_name=self.lead_name,
            annotation=self.annotation,
        )

    def with_annotation(self, annotation: ECGAnnotation | None) -> "ECGSignal":
        return ECGSignal(
            sample=self.sample,
            sample_rate=self.sample_rate,
            channel=self.channel,
            lead_name=self.lead_name,
            annotation=annotation,
        )


@dataclass(frozen=True, eq=False)
class ECGRecord:
    record_name: str | None
    signals: tuple[ECGSignal, ...]

    def __post_init__(self):
        if not self.signals:
            raise ValueError("ECGRecord must contain at least one signal")

        channels = [s.channel for s in self.signals]
        if len(channels) != len(set(channels)):
            raise ValueError("Duplicate channel numbers")

        sample_rates = {s.sample_rate for s in self.signals}
        if len(sample_rates) > 1:
            raise ValueError(
                f"All signals in a record must share the same sample_rate, got {sample_rates}"
            )

        lengths = {s.n_samples for s in self.signals}
        if len(lengths) > 1:
            raise ValueError(f"Signals have different lengths: {sorted(lengths)}")

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self) -> Iterator[ECGSignal]:
        return iter(self.signals)

    def __repr__(self) -> str:
        return (
            f"ECGRecord(record_name={self.record_name!r}, "
            f"channels={[s.channel for s in self.signals]}, "
            f"leads={self.lead_names})"
        )

    @property
    def n_samples(self) -> int:
        return self.signals[0].n_samples

    @property
    def sample_rate(self) -> int:
        return self.signals[0].sample_rate

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.signals), self.n_samples

    @property
    def lead_names(self) -> list[str | None]:
        return [s.lead_name for s in self.signals]

    def get_signal(self, channel: int) -> ECGSignal:
        signal = next((s for s in self.signals if s.channel == channel), None)
        if signal is None:
            raise KeyError(f"Signal {channel} not found")
        return signal
