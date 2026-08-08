from dataclasses import dataclass
from enum import Flag, auto

import numpy as np

from ecg_waveform.core import ECGSignal, ECGAnnotation, Fiducials

FIDUCIAL_OFFSETS = {
    Fiducials.P_WAVE: -0.20,
    Fiducials.Q_WAVE: -0.05,
    Fiducials.R_WAVE: 0.00,
    Fiducials.S_WAVE: 0.03,
    Fiducials.T_WAVE: 0.30,
}


class NoiseComponent(Flag):
    NONE = 0
    BASELINE_WANDER = auto()
    POWERLINE = auto()
    EMG = auto()
    WHITE = auto()
    ALL = BASELINE_WANDER | POWERLINE | EMG | WHITE


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    components: NoiseComponent = NoiseComponent.ALL
    noise_level: float = 1.0

    baseline_wander_amp: float = 0.15
    baseline_wander_freq: float = 0.3

    powerline_freq: float = 60.0
    powerline_amp: float = 0.02

    emg_std: float = 0.015

    white_noise_std: float = 0.005

    def has(self, component: NoiseComponent) -> bool:
        return component in self.components


def _add_gaussian_wave(
    signal: np.ndarray,
    sample_rate: float,
    center_sec: float,
    amplitude: float,
    width_sec: float,
) -> None:
    half_window = max(1, int(np.ceil(4 * width_sec * sample_rate)))
    center_idx = int(round(center_sec * sample_rate))

    start = max(0, center_idx - half_window)
    end = min(len(signal), center_idx + half_window)

    if start >= end:
        return

    local_t = (np.arange(start, end) - center_idx) / sample_rate
    signal[start:end] += amplitude * np.exp(-(local_t**2) / (2 * width_sec**2))


def _simulate_beat(
    signal: np.ndarray,
    sample_rate: float,
    r_peak_time: float,
) -> None:
    components = [
        (FIDUCIAL_OFFSETS[Fiducials.P_WAVE], 0.10, 0.025),
        (FIDUCIAL_OFFSETS[Fiducials.Q_WAVE], -0.10, 0.010),
        (FIDUCIAL_OFFSETS[Fiducials.R_WAVE], 1.20, 0.010),
        (FIDUCIAL_OFFSETS[Fiducials.S_WAVE], -0.25, 0.010),
        (FIDUCIAL_OFFSETS[Fiducials.T_WAVE], 0.30, 0.040),
    ]

    for offset, amplitude, width in components:
        _add_gaussian_wave(
            signal,
            sample_rate,
            center_sec=r_peak_time + offset,
            amplitude=amplitude,
            width_sec=width,
        )


def _add_noise(
    t: np.ndarray,
    signal: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    noisy = signal.copy()
    level = config.noise_level

    if config.has(NoiseComponent.BASELINE_WANDER):
        phase = rng.uniform(0, 2 * np.pi)
        baseline = (config.baseline_wander_amp * level) * np.sin(
            2 * np.pi * config.baseline_wander_freq * t + phase
        )
        baseline += (config.baseline_wander_amp * level * 0.3) * np.sin(
            2 * np.pi * (config.baseline_wander_freq * 2.3) * t + phase / 2
        )
        noisy += baseline

    if config.has(NoiseComponent.POWERLINE):
        noisy += (config.powerline_amp * level) * np.sin(
            2 * np.pi * config.powerline_freq * t
        )

    if config.has(NoiseComponent.EMG):
        raw_emg = rng.normal(0, config.emg_std * level, size=t.shape)
        emg = np.convolve(raw_emg, [1, -0.9], mode="same")
        noisy += emg

    if config.has(NoiseComponent.WHITE):
        noisy += rng.normal(0, config.white_noise_std * level, size=t.shape)

    return noisy


def simulate_ecg(
    duration_sec: float = 10.0,
    sample_rate: int = 200,
    heart_rate_bpm: float = 70.0,
    hrv_std_sec: float = 0.03,
    noise: NoiseConfig | None = NoiseConfig(),
    seed: int | None = 42,
) -> ECGSignal:
    rng = np.random.default_rng(seed)

    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate

    rr_mean = 60.0 / heart_rate_bpm

    r_peak_times = []
    current = rr_mean / 2

    while current < duration_sec:
        r_peak_times.append(current)
        current += rr_mean + rng.normal(0, hrv_std_sec)

    signal = np.zeros(n_samples)

    for r_time in r_peak_times:
        _simulate_beat(signal, sample_rate, r_time)

    if noise is not None and noise.components is not NoiseComponent.NONE:
        signal = _add_noise(t, signal, rng, noise)

    samples = []
    symbols = []

    for r_time in r_peak_times:
        for fiducial, offset in FIDUCIAL_OFFSETS.items():
            fid_time = r_time + offset
            sample = round(fid_time * sample_rate)

            if 0 <= sample < n_samples:
                samples.append(sample)
                symbols.append(fiducial.value)

    order = np.argsort(samples)

    annotation = ECGAnnotation(
        sample=np.asarray(samples, dtype=int)[order],
        symbol=np.asarray(symbols)[order],
    )

    return ECGSignal(
        sample=signal,
        sample_rate=sample_rate,
        channel=0,
        lead_name="synthetic",
        annotation=annotation,
    )