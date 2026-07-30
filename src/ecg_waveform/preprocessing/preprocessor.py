from collections.abc import Callable
from functools import partial
from typing import Any

from ecg_waveform.core import ECGSignal

from .denoise import (
    median_smooth,
    savgol_smooth,
    wavelet_denoise,
)
from .filter import (
    bessel_filter,
    butterworth_filter,
    chebyshev1_filter,
    chebyshev2_filter,
    elliptic_filter,
    median_baseline_filter,
    notch_filter,
)
from .normalize import (
    z_score_normalize,
)

PreprocessingStep = Callable[[ECGSignal], ECGSignal]


STEP_REGISTRY: dict[str, Callable[..., ECGSignal]] = {
    "median_smooth": median_smooth,
    "savgol_smooth": savgol_smooth,
    "wavelet_denoise": wavelet_denoise,
    "bessel_filter": bessel_filter,
    "butterworth_filter": butterworth_filter,
    "chebyshev1_filter": chebyshev1_filter,
    "chebyshev2_filter": chebyshev2_filter,
    "elliptic_filter": elliptic_filter,
    "median_baseline_filter": median_baseline_filter,
    "notch_filter": notch_filter,
    "z_score_normalize": z_score_normalize,
}


def register_step(name: str, func: Callable[..., ECGSignal]) -> None:
    if name in STEP_REGISTRY:
        raise ValueError(f"A preprocessing step named '{name}' is already registered.")
    STEP_REGISTRY[name] = func


def build_pipeline(step_configs: list[dict[str, Any]]) -> list[PreprocessingStep]:
    steps: list[PreprocessingStep] = []

    for step_config in step_configs:
        name = step_config["name"]
        params = step_config.get("params", {})

        if name not in STEP_REGISTRY:
            raise KeyError(
                f"Unknown preprocessing step '{name}'. "
                f"Available steps: {sorted(STEP_REGISTRY)}"
            )

        steps.append(partial(STEP_REGISTRY[name], **params))

    return steps


def apply_pipeline(signal: ECGSignal, steps: list[PreprocessingStep]) -> ECGSignal:
    for step in steps:
        signal = step(signal)

    return signal


class Preprocessor:
    def __init__(self, step_configs: list[dict[str, Any]]) -> None:
        self.step_configs = step_configs
        self.steps = build_pipeline(step_configs)

    def __repr__(self) -> str:
        names = [step_config["name"] for step_config in self.step_configs]
        return f"Preprocessor(steps={names})"

    def __call__(self, signal: ECGSignal) -> ECGSignal:
        return apply_pipeline(signal, self.steps)

    def run_stepwise(self, signal: ECGSignal) -> list[tuple[str, ECGSignal]]:
        stages: list[tuple[str, ECGSignal]] = [("raw", signal)]
        current = signal

        for step_config, step in zip(self.step_configs, self.steps):
            current = step(current)
            stages.append((step_config["name"], current))

        return stages

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Preprocessor":
        return cls(config["steps"])
