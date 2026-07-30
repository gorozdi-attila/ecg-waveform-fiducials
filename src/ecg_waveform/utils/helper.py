import numpy as np
import pywt
from scipy.ndimage import median_filter
from scipy.signal import spectrogram, welch

from ecg_waveform.core import ECGAnnotation, ECGSignal


def compute_window(
    n_sample: int,
    sample_rate: float,
    start_sec: float,
    interval_sec: float,
) -> tuple[int, int]:
    start = max(0, min(int(start_sec * sample_rate), n_sample))
    end = max(start, min(int((start_sec + interval_sec) * sample_rate), n_sample))

    if start == end:
        raise ValueError(
            "Requested window is outside the signal bounds or has zero length "
            f"(start={start}, end={end})."
        )

    return start, end


def compute_rr_intervals(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
) -> np.ndarray:
    if len(r_peaks.sample) < 2:
        raise ValueError("At least two R-peaks are required.")

    return np.diff(r_peaks.sample) / signal.sample_rate * 1000


def compute_baseline(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
) -> np.ndarray:
    window1 = int((window1_ms / 1000) * signal.sample_rate)
    window2 = int((window2_ms / 1000) * signal.sample_rate)

    if window1 % 2 == 0:
        window1 += 1
    if window2 % 2 == 0:
        window2 += 1

    baseline = median_filter(signal.sample, size=window1)
    baseline = median_filter(baseline, size=window2)

    return baseline


def compute_psd(
    signal: ECGSignal,
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(signal.sample) == 0:
        return np.array([]), np.array([])

    if nperseg is None:
        nperseg = min(len(signal.sample), 4096, max(256, len(signal.sample) // 4))

    freqs, psd = welch(
        signal.sample,
        fs=signal.sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    return freqs, psd


def compute_fft(
    signal: ECGSignal,
) -> tuple[np.ndarray, np.ndarray]:
    x = signal.sample.astype(float)

    x -= np.mean(x)

    window = np.hanning(len(x))
    x *= window

    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / signal.sample_rate)

    magnitude = np.abs(fft) / np.sum(window)

    if len(x) > 1:
        magnitude[1:-1] *= 2

    return freqs, magnitude


def compute_spectrogram(
    signal: ECGSignal,
    nperseg: int = 256,
    noverlap: int = 200,
    scaling: str = "density",
    mode: str = "magnitude",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if noverlap >= nperseg:
        raise ValueError("noverlap must be smaller than nperseg.")

    freqs, time, S = spectrogram(
        signal.sample,
        fs=signal.sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling=scaling,
        mode=mode,
    )

    return freqs, time, S


def compute_wavelet(
    signal: ECGSignal,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if scales is None:
        scales = np.arange(1, 128)

    coef, freqs = pywt.cwt(
        signal.sample,
        scales,
        wavelet,
        sampling_period=1 / signal.sample_rate,
    )

    return coef, freqs