from collections.abc import Callable

from ecg_waveform.core import ECGSignal

from .metrics import (
    band_power_ratio,
    baseline_wander_ratio,
    flatline_ratio,
    kurtosis_sqi,
    powerline_noise_ratio,
    qrs_power_sqi,
)

QUALITY_METRIC_REGISTRY: dict[str, Callable[..., float]] = {
    "kurtosis_sqi": kurtosis_sqi,
    "band_power_ratio": band_power_ratio,
    "qrs_power_sqi": qrs_power_sqi,
    "powerline_noise_ratio": powerline_noise_ratio,
    "baseline_wander_ratio": baseline_wander_ratio,
    "flatline_ratio": flatline_ratio,
}


def register_quality_metric(name: str, func: Callable[[ECGSignal], float]) -> None:
    if name in QUALITY_METRIC_REGISTRY:
        raise ValueError(f"A quality metric named '{name}' is already registered.")
    QUALITY_METRIC_REGISTRY[name] = func