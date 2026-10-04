from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

WORKING_RATE = 250
SCALES = (1, 2, 3, 4, 5)
LOWPASS = np.array([1.0, 3.0, 3.0, 1.0]) / 8.0
HIGHPASS = np.array([2.0, -2.0])
MIN_WORKING_SAMPLES = 2 ** (len(SCALES) + 1) * 4

DEFAULT_RMS_WINDOW_S = 2.0
DEFAULT_QRS_THRESHOLD_FACTORS = (0.5, 0.5, 0.6, 0.8)
DEFAULT_PEAK_REFINE_S = 0.012


def samples(seconds: float) -> int:
    return int(round(seconds * WORKING_RATE))


def to_working_rate(ecg: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == WORKING_RATE:
        return ecg
    divisor = gcd(WORKING_RATE, sample_rate)
    return resample_poly(ecg, WORKING_RATE // divisor, sample_rate // divisor)


def _upsample_filter(taps: np.ndarray, factor: int) -> np.ndarray:
    upsampled = np.zeros((len(taps) - 1) * factor + 1)
    upsampled[::factor] = taps
    return upsampled


def wavelet_transform(ecg: np.ndarray) -> dict[int, np.ndarray]:
    pad = 2 ** (len(SCALES) + 1)
    approximation = np.pad(ecg, pad, mode="edge")
    coefficients: dict[int, np.ndarray] = {}
    for k in SCALES:
        dilation = 2 ** (k - 1)
        detail = np.convolve(approximation, _upsample_filter(HIGHPASS, dilation))
        approximation = np.convolve(approximation, _upsample_filter(LOWPASS, dilation))
        start = pad + 2**k - 2
        coefficients[k] = detail[start : start + len(ecg)]
    return coefficients


def find_modulus_maxima(wavelet_coefficients: np.ndarray) -> np.ndarray:
    modulus = np.abs(wavelet_coefficients)
    is_maximum = (
        (modulus[1:-1] >= modulus[:-2])
        & (modulus[1:-1] > modulus[2:])
        & (modulus[1:-1] > 0)
    )
    return np.flatnonzero(is_maximum) + 1


def _moving_rms(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values**2)))
    index = np.arange(len(values))
    low = np.clip(index - window // 2, 0, len(values))
    high = np.clip(index + window // 2 + 1, 0, len(values))
    return np.sqrt((cumulative[high] - cumulative[low]) / (high - low))


@dataclass
class Analysis:
    wavelet_coefficients: dict[int, np.ndarray]
    modulus_maxima: dict[int, np.ndarray]
    local_rms: dict[int, np.ndarray]

    @classmethod
    def from_ecg(cls, ecg: np.ndarray, rms_window_s: float) -> "Analysis":
        coefficients = wavelet_transform(ecg)
        window = samples(rms_window_s)
        return cls(
            wavelet_coefficients=coefficients,
            modulus_maxima={k: find_modulus_maxima(w) for k, w in coefficients.items()},
            local_rms={k: _moving_rms(w, window) for k, w in coefficients.items()},
        )

    @property
    def n_samples(self) -> int:
        return len(self.wavelet_coefficients[1])

    def thresholds(
        self, factors: tuple[float, ...], scale_factor: float = 1.0
    ) -> dict[int, np.ndarray]:
        return {
            k: scale_factor * factor * self.local_rms[k]
            for k, factor in enumerate(factors, start=1)
        }


def zero_crossings(
    coefficients: np.ndarray, low: int, high: int, falling: bool
) -> list[tuple[int, float]]:
    crossings = []
    for n in range(max(low, 0), min(high, len(coefficients) - 1)):
        a, b = coefficients[n], coefficients[n + 1]
        hit = (a > 0 and b <= 0) if falling else (a < 0 and b >= 0)
        if hit:
            crossings.append((n, n - 0.5 + a / (a - b)))
    return crossings


def significant_crossings(
    analysis: Analysis,
    thresholds: dict[int, np.ndarray],
    scale: int,
    low: int,
    high: int,
    falling: bool,
) -> list[float]:
    coefficients = analysis.wavelet_coefficients[scale]
    maxima = analysis.modulus_maxima[scale]
    threshold = thresholds[scale]
    locations = []
    for n, location in zero_crossings(coefficients, low, high, falling):
        right = int(np.searchsorted(maxima, n, side="right"))
        if right == 0 or right >= len(maxima):
            continue
        left_max, right_max = maxima[right - 1], maxima[right]
        if (
            abs(coefficients[left_max]) >= threshold[left_max]
            and abs(coefficients[right_max]) >= threshold[right_max]
        ):
            locations.append(location)
    return locations


def threshold_crossing(
    coefficients: np.ndarray, start: int, step: int, fraction: float, limit: int
) -> int:
    modulus = np.abs(coefficients)
    threshold = fraction * modulus[start]
    n = start
    for _ in range(limit):
        following = n + step
        if following < 0 or following >= len(coefficients):
            break
        if modulus[following] < threshold:
            return following
        if modulus[following] > modulus[n]:
            return n
        n = following
    return n


def link_maxima(
    analysis: Analysis,
    start_scale: int,
    start_position: int,
    thresholds: dict[int, np.ndarray],
) -> dict[int, int]:
    coefficients = analysis.wavelet_coefficients
    sign = np.sign(coefficients[start_scale][start_position])
    line = {start_scale: start_position}
    current = start_position
    for scale in range(start_scale - 1, 0, -1):
        maxima = analysis.modulus_maxima[scale]
        near = maxima[np.abs(maxima - current) <= 2**scale + 2]
        near = [
            int(p)
            for p in near
            if np.sign(coefficients[scale][p]) == sign
            and abs(coefficients[scale][p]) >= thresholds[scale][p]
        ]
        if near:
            current = min(near, key=lambda p: abs(p - current))
            line[scale] = current
    return line


def to_original_sample(position: float, rate_ratio: float, n_samples: int) -> int:
    return int(np.clip(round(position * rate_ratio), 0, n_samples - 1))


def localize_peak(
    original: np.ndarray,
    position: float,
    is_maximum: bool,
    rate_ratio: float,
    refine_s: float,
) -> int:
    radius = max(1, int(round(refine_s * WORKING_RATE * rate_ratio)))
    center = to_original_sample(position, rate_ratio, len(original))
    low, high = max(0, center - radius), min(len(original), center + radius + 1)
    segment = original[low:high]
    return low + int(np.argmax(segment) if is_maximum else np.argmin(segment))
