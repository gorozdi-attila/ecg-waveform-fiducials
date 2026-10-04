from dataclasses import dataclass
from enum import Flag, auto

import numpy as np

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials


QT_REFERENCE_RR = 60.0 / 70.0


@dataclass(frozen=True, slots=True)
class Wave:
    center: float
    amplitude: float
    width_up: float
    width_down: float
    onset: float
    offset: float

    def __post_init__(self) -> None:
        if self.width_up <= 0:
            raise ValueError("width_up must be positive")

        if self.width_down <= 0:
            raise ValueError("width_down must be positive")

        if self.onset > self.center:
            raise ValueError("onset must not occur after center")

        if self.offset < self.center:
            raise ValueError("offset must not occur before center")


WAVES = {
    "P": Wave(
        center=-0.20,
        amplitude=0.10,
        width_up=0.020,
        width_down=0.030,
        onset=-0.28,
        offset=-0.12,
    ),
    "Q": Wave(
        center=-0.05,
        amplitude=-0.10,
        width_up=0.010,
        width_down=0.010,
        onset=-0.07,
        offset=-0.03,
    ),
    "R": Wave(
        center=0.00,
        amplitude=1.20,
        width_up=0.010,
        width_down=0.010,
        onset=-0.01,
        offset=0.01,
    ),
    "S": Wave(
        center=0.03,
        amplitude=-0.25,
        width_up=0.010,
        width_down=0.010,
        onset=0.01,
        offset=0.08,
    ),
    "T": Wave(
        center=0.30,
        amplitude=0.30,
        width_up=0.030,
        width_down=0.055,
        onset=0.20,
        offset=0.40,
    ),
}

_QT_SCALED_WAVES = {"T"}

_QT_SCALED_FIDUCIALS = {Fiducials.T_PEAK, Fiducials.T_ONSET, Fiducials.T_OFFSET}

_FIDUCIAL_WAVE_MAP = {
    Fiducials.P_ONSET: ("P", "onset"),
    Fiducials.P_PEAK: ("P", "center"),
    Fiducials.P_OFFSET: ("P", "offset"),
    Fiducials.QRS_ONSET: ("Q", "onset"),
    Fiducials.Q_PEAK: ("Q", "center"),
    Fiducials.R_PEAK: ("R", "center"),
    Fiducials.S_PEAK: ("S", "center"),
    Fiducials.QRS_OFFSET: ("S", "offset"),
    Fiducials.T_ONSET: ("T", "onset"),
    Fiducials.T_PEAK: ("T", "center"),
    Fiducials.T_OFFSET: ("T", "offset"),
}


class NoiseComponent(Flag):
    BASELINE_WANDER = auto()
    POWERLINE_INTERFERENCE = auto()
    MUSCLE_NOISE = auto()
    ELECTRODE_MOTION_ARTIFACT = auto()
    MOTION_ARTIFACT = auto()
    WHITE_NOISE = auto()
    SATURATION = auto()

    ALL = (
        BASELINE_WANDER
        | POWERLINE_INTERFERENCE
        | MUSCLE_NOISE
        | ELECTRODE_MOTION_ARTIFACT
        | MOTION_ARTIFACT
        | WHITE_NOISE
        | SATURATION
    )


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_range(
    name: str,
    value: tuple[float, float],
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")

    low, high = value

    if low > high:
        raise ValueError(f"{name} must satisfy low <= high")

    if strictly_positive:
        if low <= 0 or high <= 0:
            raise ValueError(f"{name} must contain positive values")
    elif minimum is not None:
        if low < minimum:
            raise ValueError(f"{name} values must be >= {minimum}")


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    components: NoiseComponent = NoiseComponent.ALL
    noise_level: float = 1.0

    baseline_wander_amplitude: float = 0.20
    baseline_wander_frequency: float = 0.25

    powerline_frequency: float = 60.0
    powerline_amplitude: float = 0.02

    emg_std: float = 0.015

    white_noise_std: float = 0.005

    electrode_motion_probability: float = 0.20
    electrode_motion_amplitude: tuple[float, float] = (0.20, 1.00)
    electrode_motion_duration: tuple[float, float] = (0.10, 1.00)

    motion_event_probability: float = 0.15
    motion_amplitude: float = 0.15
    motion_frequency: tuple[float, float] = (0.1, 2.0)
    motion_event_duration: tuple[float, float] = (0.5, 3.0)

    saturation_threshold: float = 1.5

    def __post_init__(self) -> None:
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative")

        if self.baseline_wander_amplitude < 0:
            raise ValueError("baseline_wander_amplitude must be non-negative")

        if self.baseline_wander_frequency < 0:
            raise ValueError("baseline_wander_frequency must be non-negative")

        if self.powerline_frequency < 0:
            raise ValueError("powerline_frequency must be non-negative")

        if self.powerline_amplitude < 0:
            raise ValueError("powerline_amplitude must be non-negative")

        if self.emg_std < 0:
            raise ValueError("emg_std must be non-negative")

        if self.white_noise_std < 0:
            raise ValueError("white_noise_std must be non-negative")

        if self.saturation_threshold <= 0:
            raise ValueError("saturation_threshold must be positive")

        _validate_probability(
            "electrode_motion_probability",
            self.electrode_motion_probability,
        )

        _validate_probability(
            "motion_event_probability",
            self.motion_event_probability,
        )

        _validate_range(
            "electrode_motion_amplitude",
            self.electrode_motion_amplitude,
            minimum=0.0,
        )

        _validate_range(
            "electrode_motion_duration",
            self.electrode_motion_duration,
            minimum=0.0,
            strictly_positive=True,
        )

        _validate_range(
            "motion_frequency",
            self.motion_frequency,
            minimum=0.0,
        )

        _validate_range(
            "motion_event_duration",
            self.motion_event_duration,
            minimum=0.0,
            strictly_positive=True,
        )

        if self.motion_amplitude < 0:
            raise ValueError("motion_amplitude must be non-negative")

    def has(self, component: NoiseComponent) -> bool:
        return bool(self.components & component)


def _add_baseline_wander(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> None:
    amplitude = config.baseline_wander_amplitude * config.noise_level

    phase_1 = rng.uniform(0.0, 2.0 * np.pi)
    phase_2 = rng.uniform(0.0, 2.0 * np.pi)

    baseline = amplitude * np.sin(
        2.0 * np.pi * config.baseline_wander_frequency * time + phase_1
    ) + 0.3 * amplitude * np.sin(
        2.0 * np.pi * config.baseline_wander_frequency * 2.3 * time + phase_2
    )

    signal += baseline


def _add_powerline_interference(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> None:
    signal += (config.powerline_amplitude * config.noise_level) * np.sin(
        2 * np.pi * config.powerline_frequency * time
    )


def _add_muscle_noise(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> None:
    emg = rng.normal(0, config.emg_std * config.noise_level, size=time.shape)

    filter_taps = np.array([1.0, -0.9])
    filter_gain = np.sqrt(np.sum(filter_taps**2))

    signal += np.convolve(emg, filter_taps / filter_gain, mode="same")


def _add_electrode_motion_artifact(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> None:
    duration = time[-1]

    if duration <= 0:
        return

    n_opportunities = max(
        1,
        int(np.ceil(duration / 2.0)),
    )

    for _ in range(n_opportunities):
        if rng.random() > config.electrode_motion_probability:
            continue

        center = rng.uniform(
            0.0,
            duration,
        )

        amplitude = rng.uniform(*config.electrode_motion_amplitude)

        amplitude *= config.noise_level

        amplitude *= rng.choice([-1.0, 1.0])

        artifact_duration = rng.uniform(*config.electrode_motion_duration)

        rise = artifact_duration * rng.uniform(
            0.15,
            0.35,
        )

        decay = artifact_duration * rng.uniform(
            0.5,
            1.5,
        )

        dt = time - center

        artifact = np.zeros_like(time)

        mask = dt >= 0.0

        artifact[mask] = amplitude * (
            np.exp(-dt[mask] / decay) - np.exp(-dt[mask] / rise)
        )

        max_abs = np.max(np.abs(artifact))

        if max_abs > 0:
            artifact *= abs(amplitude) / max_abs

        signal += artifact


def _add_motion_artifact(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> None:
    duration = time[-1]

    if duration <= 0:
        return

    n_opportunities = max(
        1,
        int(np.ceil(duration / 3.0)),
    )

    for _ in range(n_opportunities):
        if rng.random() > config.motion_event_probability:
            continue

        center = rng.uniform(
            0.0,
            duration,
        )

        event_duration = rng.uniform(*config.motion_event_duration)

        start = max(
            0.0,
            center - event_duration / 2.0,
        )

        end = min(
            duration,
            center + event_duration / 2.0,
        )

        mask = (time >= start) & (time <= end)

        if not np.any(mask):
            continue

        local_time = time[mask] - center

        sigma = event_duration / 3.0

        envelope = np.exp(-0.5 * (local_time / sigma) ** 2)

        frequency = rng.uniform(*config.motion_frequency)

        phase = rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        oscillation = np.sin(2.0 * np.pi * frequency * local_time + phase)

        random_component = rng.normal(
            0.0,
            0.2,
            size=np.count_nonzero(mask),
        )

        artifact = (
            config.motion_amplitude
            * config.noise_level
            * envelope
            * (0.7 * oscillation + 0.3 * random_component)
        )

        signal[mask] += artifact


def _add_white_noise(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> None:
    signal += rng.normal(
        0,
        config.white_noise_std * config.noise_level,
        size=time.shape,
    )


def _apply_saturation(
    signal: np.ndarray,
    config: NoiseConfig,
) -> None:
    threshold = config.saturation_threshold

    np.clip(
        signal,
        -threshold,
        threshold,
        out=signal,
    )


_NOISE_COMPONENT_HANDLER = {
    NoiseComponent.BASELINE_WANDER: _add_baseline_wander,
    NoiseComponent.POWERLINE_INTERFERENCE: _add_powerline_interference,
    NoiseComponent.MUSCLE_NOISE: _add_muscle_noise,
    NoiseComponent.ELECTRODE_MOTION_ARTIFACT: _add_electrode_motion_artifact,
    NoiseComponent.MOTION_ARTIFACT: _add_motion_artifact,
    NoiseComponent.WHITE_NOISE: _add_white_noise,
}


def _add_noise(
    signal: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:

    for component, handler in _NOISE_COMPONENT_HANDLER.items():
        if config.has(component):
            handler(
                signal,
                time,
                rng,
                config,
            )

    if config.has(NoiseComponent.SATURATION):
        _apply_saturation(
            signal,
            config,
        )

    return signal


def _qt_scale_factor(rr: float) -> float:
    return np.sqrt(rr / QT_REFERENCE_RR)


def _add_asymmetric_gaussian_inplace(
    signal: np.ndarray,
    time: np.ndarray,
    center: float,
    amplitude: float,
    width_up: float,
    width_down: float,
) -> None:
    if width_up <= 0.0 or width_down <= 0.0:
        raise ValueError("width_up and width_down must be positive")

    left = max(0, np.searchsorted(time, center - 4.0 * width_up, side="left"))
    right = min(
        signal.size,
        np.searchsorted(time, center + 4.0 * width_down, side="right"),
    )

    if left >= right:
        return

    t = time[left:right]
    dt = t - center

    width = np.where(dt < 0.0, width_up, width_down)

    signal[left:right] += amplitude * np.exp(-0.5 * (dt / width) ** 2)


def _generate_rr_intervals(
    duration_sec: float,
    mean_rr: float,
    hrv_std_sec: float,
    rng: np.random.Generator,
    initial_time: float,
) -> tuple[list[float], list[float]]:
    beat_times: list[float] = []
    beat_rrs: list[float] = []

    beat_time = initial_time
    previous_rr = mean_rr

    while beat_time < duration_sec:
        beat_times.append(beat_time)
        beat_rrs.append(previous_rr)

        rr = mean_rr + rng.normal(
            0.0,
            hrv_std_sec,
        )

        rr = max(rr, 0.30)

        beat_time += rr
        previous_rr = rr

    return beat_times, beat_rrs


def _multiplicative_jitter(
    rng: np.random.Generator,
    std: float,
) -> float:
    if std == 0.0:
        return 1.0

    return float(np.exp(rng.normal(0.0, std)))


def _wave_timing(
    wave: Wave,
    beat_time: float,
    qt_scale: float,
    *,
    scale_qt: bool,
) -> tuple[float, float, float]:
    if scale_qt:
        return (
            beat_time + wave.center * qt_scale,
            wave.width_up * qt_scale,
            wave.width_down * qt_scale,
        )

    return (
        beat_time + wave.center,
        wave.width_up,
        wave.width_down,
    )


def _fiducial_offset(
    fiducial: Fiducials,
    qt_scale: float,
) -> float:
    wave_name, attribute = _FIDUCIAL_WAVE_MAP[fiducial]
    wave = WAVES[wave_name]

    offset = float(getattr(wave, attribute))

    if fiducial in _QT_SCALED_FIDUCIALS:
        return offset * qt_scale

    return offset


def simulate_ecg(
    duration_sec: float = 10.0,
    sample_rate: int = 200,
    heart_rate_bpm: float = 70.0,
    hrv_std_sec: float = 0.03,
    respiration_rate_hz: float = 0.25,
    respiration_amplitude_modulation: float = 0.05,
    morphology_jitter_std: float = 0.03,
    noise: NoiseConfig | None = NoiseConfig(),
    seed: int | None = 42,
) -> ECGSignal:
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("sample_rate must be an integer")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if heart_rate_bpm <= 0.0:
        raise ValueError("heart_rate_bpm must be positive")

    if hrv_std_sec < 0.0:
        raise ValueError("hrv_std_sec must be non-negative")

    if respiration_rate_hz < 0.0:
        raise ValueError("respiration_rate_hz must be non-negative")

    if not 0.0 <= respiration_amplitude_modulation < 1.0:
        raise ValueError("respiration_amplitude_modulation must be in [0, 1)")

    if morphology_jitter_std < 0.0:
        raise ValueError("morphology_jitter_std must be non-negative")

    if (
        noise is not None
        and noise.has(NoiseComponent.POWERLINE_INTERFERENCE)
        and noise.powerline_frequency >= sample_rate / 2.0
    ):
        raise ValueError("powerline_frequency must be below the Nyquist frequency")

    rng = np.random.default_rng(seed)

    n_samples = int(round(duration_sec * sample_rate))

    if n_samples < 1:
        raise ValueError(
            "duration_sec and sample_rate must produce at least one sample"
        )

    time = np.arange(n_samples, dtype=np.float64) / sample_rate

    mean_rr = 60.0 / heart_rate_bpm

    signal = np.zeros(
        n_samples,
        dtype=np.float64,
    )

    points: dict[Fiducials, list[int]] = {fiducial: [] for fiducial in Fiducials}

    respiration_phase = rng.uniform(
        0.0,
        2.0 * np.pi,
    )

    beat_times, beat_rrs = _generate_rr_intervals(
        duration_sec,
        mean_rr,
        hrv_std_sec,
        rng,
        initial_time=0.5,
    )

    for beat_time, rr in zip(beat_times, beat_rrs):
        qt_scale = _qt_scale_factor(rr)

        resp_mod = 1.0 + (
            respiration_amplitude_modulation
            * np.sin(2.0 * np.pi * respiration_rate_hz * beat_time + respiration_phase)
        )

        for wave_name, wave in WAVES.items():
            wave_center, wave_width_up, wave_width_down = _wave_timing(
                wave,
                beat_time,
                qt_scale,
                scale_qt=wave_name in _QT_SCALED_WAVES,
            )

            amplitude_jitter = _multiplicative_jitter(
                rng,
                morphology_jitter_std,
            )

            width_jitter = _multiplicative_jitter(
                rng,
                morphology_jitter_std,
            )

            beat_amplitude = wave.amplitude * amplitude_jitter * resp_mod

            _add_asymmetric_gaussian_inplace(
                signal,
                time,
                wave_center,
                beat_amplitude,
                wave_width_up * width_jitter,
                wave_width_down * width_jitter,
            )

        for fiducial in Fiducials:
            offset = _fiducial_offset(
                fiducial,
                qt_scale,
            )

            fiducial_time = beat_time + offset

            sample = int(round(fiducial_time * sample_rate))

            if 0 <= sample < n_samples:
                points[fiducial].append(sample)

    annotation_points = {
        fiducial: np.asarray(
            samples,
            dtype=np.int64,
        )
        for fiducial, samples in points.items()
    }

    annotation = ECGAnnotation.from_points(annotation_points)

    if noise is not None:
        _add_noise(
            signal=signal,
            time=time,
            rng=rng,
            config=noise,
        )

    return ECGSignal(
        sample=signal,
        sample_rate=sample_rate,
        channel=0,
        lead_name="synthetic",
        annotation=annotation,
    )
