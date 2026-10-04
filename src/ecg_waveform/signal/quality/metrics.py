from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis

from ecg_waveform.core import ECGSignal
from ecg_waveform.utils import compute_baseline, compute_psd

from .base import SignalQualityMetric
from .registry import register_quality_metric


def kurtosis_sqi(
    signal: ECGSignal,
    fisher: bool = True,
) -> float:
    x = np.asarray(signal.sample, dtype=float)

    if x.size == 0 or np.std(x) == 0:
        return float("nan")

    return float(kurtosis(x, fisher=fisher, bias=True))


def band_power_ratio(
    signal: ECGSignal,
    band: tuple[float, float | None],
    reference_band: tuple[float, float | None] = (0.0, None),
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> float:
    freqs, psd = compute_psd(
        signal,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )

    if freqs.size == 0:
        return float("nan")

    def _band_power(low: float, high: float | None) -> float:
        high = freqs[-1] if high is None else high
        mask = (freqs >= low) & (freqs <= high)

        if not np.any(mask):
            return 0.0

        return float(np.trapezoid(psd[mask], freqs[mask]))

    if _band_power(*reference_band) == 0.0:
        return float("nan")

    return _band_power(*band) / _band_power(*reference_band)


def qrs_power_sqi(
    signal: ECGSignal,
    qrs_band: tuple[float, float] = (5.0, 15.0),
    reference_band: tuple[float, float] = (0.5, 40.0),
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> float:
    return band_power_ratio(
        signal,
        band=qrs_band,
        reference_band=reference_band,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )


def powerline_noise_ratio(
    signal: ECGSignal,
    powerline_freq: float = 50.0,
    bandwidth: float = 1.0,
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> float:
    band = (powerline_freq - bandwidth, powerline_freq + bandwidth)

    return band_power_ratio(
        signal,
        band=band,
        reference_band=(0.0, None),
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )


def baseline_wander_ratio(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
) -> float:
    x = np.asarray(signal.sample, dtype=float)

    if x.size == 0:
        return float("nan")

    signal_power = float(np.mean(x**2))
    if signal_power == 0.0:
        return float("nan")

    baseline = compute_baseline(signal, window1_ms=window1_ms, window2_ms=window2_ms)
    baseline_power = float(np.mean(np.asarray(baseline, dtype=float) ** 2))

    return baseline_power / signal_power


def flatline_ratio(
    signal: ECGSignal, slope_threshold: float = 1e-4, min_run_ms: float = 500.0
) -> float:
    x = np.asarray(signal.sample, dtype=float)

    if x.size < 2:
        return 0.0

    min_run_samples = max(1, round(min_run_ms / 1000.0 * signal.sample_rate))

    dt = 1.0 / signal.sample_rate
    slope = np.abs(np.diff(x, prepend=x[0])) / dt
    is_flat = slope < slope_threshold

    change_points = np.flatnonzero(np.diff(is_flat.astype(np.int8))) + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [is_flat.size]))
    run_is_flat = is_flat[run_starts]
    run_lengths = run_ends - run_starts

    qualifying = run_is_flat & (run_lengths >= min_run_samples)
    flat_samples = int(run_lengths[qualifying].sum())

    return flat_samples / x.size


@register_quality_metric("kurtosis_sqi")
@dataclass(frozen=True, slots=True)
class KurtosisSQI(SignalQualityMetric):
    fisher: bool = True

    def compute(self, signal: ECGSignal) -> float:
        return kurtosis_sqi(signal, fisher=self.fisher)


@register_quality_metric("band_power_ratio")
@dataclass(frozen=True, slots=True)
class BandPowerRatioSQI(SignalQualityMetric):
    band: tuple[float, float | None]
    reference_band: tuple[float, float | None] = (0.0, None)
    nperseg: int | None = None
    window: str = "hann"
    noverlap: int | None = None

    def compute(self, signal: ECGSignal) -> float:
        return band_power_ratio(
            signal,
            band=self.band,
            reference_band=self.reference_band,
            nperseg=self.nperseg,
            window=self.window,
            noverlap=self.noverlap,
        )


@register_quality_metric("qrs_power_sqi")
@dataclass(frozen=True, slots=True)
class QRSPowerSQI(SignalQualityMetric):
    qrs_band: tuple[float, float] = (5.0, 15.0)
    reference_band: tuple[float, float] = (0.5, 40.0)
    nperseg: int | None = None
    window: str = "hann"
    noverlap: int | None = None

    def compute(self, signal: ECGSignal) -> float:
        return qrs_power_sqi(
            signal,
            qrs_band=self.qrs_band,
            reference_band=self.reference_band,
            nperseg=self.nperseg,
            window=self.window,
            noverlap=self.noverlap,
        )


@register_quality_metric("powerline_noise_ratio")
@dataclass(frozen=True, slots=True)
class PowerlineNoiseRatioSQI(SignalQualityMetric):
    powerline_freq: float = 50.0
    bandwidth: float = 1.0
    nperseg: int | None = None
    window: str = "hann"
    noverlap: int | None = None

    def compute(self, signal: ECGSignal) -> float:
        return powerline_noise_ratio(
            signal,
            powerline_freq=self.powerline_freq,
            bandwidth=self.bandwidth,
            nperseg=self.nperseg,
            window=self.window,
            noverlap=self.noverlap,
        )


@register_quality_metric("baseline_wander_ratio")
@dataclass(frozen=True, slots=True)
class BaselineWanderRatioSQI(SignalQualityMetric):
    window1_ms: int = 200
    window2_ms: int = 600

    def compute(self, signal: ECGSignal) -> float:
        return baseline_wander_ratio(
            signal, window1_ms=self.window1_ms, window2_ms=self.window2_ms
        )


@register_quality_metric("flatline_ratio")
@dataclass(frozen=True, slots=True)
class FlatlineRatioSQI(SignalQualityMetric):
    slope_threshold: float = 1e-4
    min_run_ms: float = 500.0

    def compute(self, signal: ECGSignal) -> float:
        return flatline_ratio(
            signal, slope_threshold=self.slope_threshold, min_run_ms=self.min_run_ms
        )
