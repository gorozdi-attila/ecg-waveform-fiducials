import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter
from scipy.signal import spectrogram, welch

import pywt

from ecg_waveform_fiducials.core import ECGSignal, ECGAnnotation

def _compute_window(
    n_sample: int,
    sample_rate: float,
    start_sec: float,
    interval_sec: float,
) -> tuple[int, int]:
    start = max(0, min(int(start_sec * sample_rate), n_sample))
    end = max(start, min(int((start_sec + interval_sec) * sample_rate), n_sample))

    return start, end


def _compute_baseline(
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


def _compute_psd(
    signal: ECGSignal,
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(signal.sample) == 0:
        return np.array([]), np.array([])

    if nperseg is None:
        nperseg = min(len(signal.sample), 4096, max(256, len(signal.sample) // 4))

    freqs, psd = welch(signal.sample, fs=signal.sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)

    return freqs, psd


def plot_signal(
    signal: ECGSignal,
    start_sec: float = 0,
    interval_sec: float = 10,
    show_annotation: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    start, end = _compute_window(len(signal), signal.sample_rate, start_sec, interval_sec)

    segment = signal.segment(start, end)
    
    window_start = start / signal.sample_rate
    time = segment.time + window_start

    fig, ax = plt.subplots()
    ax.plot(time, segment.sample, label="ECG Signal")

    if show_annotation and segment.annotation is not None:
        ann_x = segment.annotation.sample / segment.sample_rate + window_start
        ann_y = segment.sample[segment.annotation.sample]

        ax.scatter(ann_x, ann_y, s=30, color="red", zorder=3, label="Annotated Points")

        offset = 0.05 * (segment.sample.max() - segment.sample.min())

        for x, y, sym in zip(ann_x, ann_y, segment.annotation.symbol):
            ax.text(x, y + offset, sym, fontsize=8, ha="center", weight="bold")

    ax.set_title(f"ECG Signal — {interval_sec:.2f} s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

    plt.tight_layout()

    return fig, ax


def plot_signal_amplitude_distribution(
    signal: ECGSignal,
    physiological_limit: float = 5.0,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    sns.histplot(signal.sample, bins=30, kde=True, linewidth=0, ax=ax, label="Amplitude distribution")

    ax.axvline(-physiological_limit, color="red", linestyle="--", label=f"Min threshold ({-physiological_limit} mV)")
    ax.axvline( physiological_limit, color="red", linestyle="--", label=f"Max threshold ({ physiological_limit} mV)")

    ax.set_title("Amplitude distribution")
    ax.set_xlabel("Amplitude (mV)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

    plt.tight_layout()
    
    return fig, ax


def plot_beat_overlays(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
    window_ms: tuple[int, int] = (200, 400),
) -> tuple[plt.Figure, plt.Axes]:
    left = int(window_ms[0] * signal.sample_rate / 1000)
    right = int(window_ms[1] * signal.sample_rate / 1000)

    beats = [
        signal.sample[r - left : r + right]
        for r in r_peaks.sample
        if r - left >= 0 and r + right <= len(signal)
    ]

    if not beats:
        raise ValueError("No valid beats found in the given window.")

    beats_arr = np.asarray(beats)

    mean_beat = beats_arr.mean(axis=0)
    std_beat = beats_arr.std(axis=0)

    time = (np.arange(beats_arr.shape[1]) - left) / signal.sample_rate * 1000

    fig, ax = plt.subplots()

    ax.plot(time, mean_beat, label="Mean beat")
    ax.fill_between(
        time,
        mean_beat - std_beat,
        mean_beat + std_beat,
        alpha=0.25,
        label="±1 SD",
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="R-peak")

    ax.set_title(f"ECG Beat Overlay ({len(beats_arr)} beats)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

    plt.tight_layout()

    return fig, ax


def plot_rr_tachogram(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
) -> tuple[plt.Figure, plt.Axes]:
    if len(r_peaks.sample) < 2:
        raise ValueError("At least two R-peaks are required.")

    rr_intervals = np.diff(r_peaks.sample) / signal.sample_rate * 1000 
    beat_idx = np.arange(1, len(r_peaks.sample))

    fig, ax = plt.subplots()
    ax.plot(beat_idx, rr_intervals, "-o", markersize=3, label="RR interval")

    ax.set_title("RR Tachogram")
    ax.set_xlabel("Beat index")
    ax.set_ylabel("RR interval (ms)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

    return fig, ax


def plot_hrv(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
) -> tuple[plt.Figure, plt.Axes]:
    if len(r_peaks.sample) < 2:
        raise ValueError("At least two R-peaks are required.")
    rr_intervals = np.diff(r_peaks.sample) / signal.sample_rate * 1000

    fig, ax = plt.subplots()
    sns.histplot(rr_intervals, bins=30, kde=True, linewidth=0, ax=ax, label="RR interval distribution")

    ax.set_title(f"HRV Distribution — Mean: {np.mean(rr_intervals):.2f} ms, Std: {np.std(rr_intervals):.2f} ms")
    ax.set_xlabel("RR Interval (ms)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    plt.tight_layout()

    return fig, ax


def plot_poincare(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
) -> tuple[plt.Figure, plt.Axes]:
    if len(r_peaks.sample) < 2:
        raise ValueError("Need at least two R-peaks.")

    rr_intervals = np.diff(r_peaks.sample) / signal.sample_rate * 1000

    fig, ax = plt.subplots()
    ax.scatter(rr_intervals[:-1], rr_intervals[1:], s=30, label="RRₙ vs RRₙ₊₁")

    min_rr, max_rr = np.min(rr_intervals), np.max(rr_intervals)
    ax.plot([min_rr, max_rr], [min_rr, max_rr], "r--", linewidth=1)

    ax.set_title("Poincaré Plot")
    ax.set_xlabel("RRₙ (ms)")
    ax.set_ylabel("RRₙ₊₁ (ms)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax.set_aspect("auto")
    plt.tight_layout()

    return fig, ax


def plot_spectrogram(
    signal: ECGSignal,
    start_sec: float = 0,
    interval_sec: float = 10,
) -> tuple[plt.Figure, plt.Axes]:
    start, end = _compute_window(len(signal), signal.sample_rate, start_sec, interval_sec)
    segment = signal.segment(start, end)

    f, t, S = spectrogram(segment.sample, segment.sample_rate, nperseg=256, noverlap=200, scaling="density", mode="magnitude")
    S_db = 10 * np.log10(np.maximum(S, 1e-12))
    t = t + start_sec

    fig, ax = plt.subplots()

    im = ax.pcolormesh(t, f, S_db, shading="auto", cmap="cividis")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()

    return fig, ax


def plot_wavelet_scalogram(
    signal: ECGSignal,
    wavelet: str = "morl",
    start_sec: float = 0,
    interval_sec: float = 10,
) -> tuple[plt.Figure, plt.Axes]:
    start, end = _compute_window(len(signal), signal.sample_rate, start_sec, interval_sec)
    segment = signal.segment(start, end)
    time = segment.time

    scales = np.arange(1, 128)
    coef, freqs = pywt.cwt(segment.sample, scales, wavelet, sampling_period=1 / segment.sample_rate)
    power = np.abs(coef)

    fig, ax = plt.subplots()
    im = ax.imshow(power, extent=[time[0], time[-1], freqs[-1], freqs[0]], aspect="auto", cmap="cividis", origin="upper")
    fig.colorbar(im, ax=ax, label="Magnitude")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Wavelet Scalogram ({wavelet})")
    plt.tight_layout()

    return fig, ax


def plot_baseline_wander(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
    start_sec: float = 0,
    interval_sec: float = 10,
) -> tuple[plt.Figure, plt.Axes]:
    start, end = _compute_window(len(signal), signal.sample_rate, start_sec, interval_sec)

    baseline = _compute_baseline(signal, window1_ms=window1_ms, window2_ms=window2_ms)
    baseline = baseline[start:end]

    time = signal.segment(start, end).time

    fig, ax = plt.subplots()

    ax.plot(time, baseline, label="Estimated Baseline", color="red")

    ax.set_title("Baseline Wander")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    plt.tight_layout()

    return fig, ax


def plot_fft(
    signal: ECGSignal,
) -> tuple[plt.Figure, plt.Axes]:
    x = signal.sample.astype(float)

    x = x - np.mean(x)

    window = np.hanning(len(x))
    x = x * window

    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / signal.sample_rate)

    magnitude = np.abs(fft) / np.sum(window)
    if len(x) > 1:
        magnitude[1:-1] *= 2

    fig, ax = plt.subplots()
    ax.plot(freqs, magnitude)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("FFT Magnitude Spectrum")

    fig.tight_layout()

    return fig, ax


def plot_psd(
    signal: ECGSignal,
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    freqs, psd = _compute_psd(signal, nperseg=nperseg, window=window, noverlap=noverlap)

    fig, ax = plt.subplots()

    ax.semilogy(freqs, psd)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (mV²/Hz)")
    ax.set_title("Power Spectral Density")

    plt.tight_layout()

    return fig, ax