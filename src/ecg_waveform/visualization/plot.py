from collections.abc import Callable
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials
from ecg_waveform.utils import (
    compute_baseline,
    compute_fft,
    compute_psd,
    compute_rr_intervals,
    compute_spectrogram,
    compute_wavelet,
    compute_window,
)


def _get_axes(
    ax: plt.Axes | None,
) -> tuple[plt.Figure, plt.Axes, bool]:
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax, True

    return ax.figure, ax, False


def _set_title(ax: plt.Axes, title: str | None, default: str) -> None:
    ax.set_title(title if title is not None else default)


def _get_segment(
    signal: ECGSignal,
    start_sec: float,
    interval_sec: float,
) -> tuple[ECGSignal, np.ndarray, int, int]:
    start, end = compute_window(
        len(signal),
        signal.sample_rate,
        start_sec,
        interval_sec,
    )

    segment = signal.segment(start, end)
    time = segment.time + start / signal.sample_rate

    return segment, time, start, end


def with_axes(plot_fn: Callable) -> Callable:
    @wraps(plot_fn)
    def wrapper(*args, ax: plt.Axes | None = None, **kwargs):
        fig, ax, created = _get_axes(ax)

        ax.grid(which="major", linewidth=0.8, color="lightgray")
        ax.grid(which="minor", linewidth=0.3, color="lightgray")
        ax.minorticks_on()

        plot_fn(*args, ax=ax, **kwargs)

        if created:
            fig.tight_layout()

        return fig, ax

    return wrapper


def _plot_annotations(
    ax: plt.Axes,
    samples: np.ndarray,
    annotation: ECGAnnotation,
    sample_rate: float,
    window_start: float,
    offset: float,
    color: str,
    label: str,
    marker: str = "o",
):
    if annotation is None or len(annotation.sample) == 0:
        return

    x = annotation.sample / sample_rate + window_start
    y = samples[annotation.sample]

    ax.scatter(
        x,
        y,
        s=30,
        marker=marker,
        color=color,
        zorder=3,
        label=label,
    )

    for xi, yi, sym in zip(x, y, annotation.symbol):
        ax.text(
            xi,
            yi + offset,
            sym,
            color=color,
            fontsize=8,
            ha="center",
            weight="bold",
        )


@with_axes
def plot_signal(
    signal: ECGSignal,
    start_sec: float = 0,
    interval_sec: float = 5,
    show_annotation: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    segment, time, start, end = _get_segment(
        signal,
        start_sec,
        interval_sec,
    )

    window_start = start / signal.sample_rate

    plot_kwargs.setdefault("label", "ECG Signal")

    ax.plot(
        time,
        segment.sample,
        **plot_kwargs,
    )

    if show_annotation and signal.annotation is not None:
        _plot_annotations(
            ax,
            segment.sample,
            segment.annotation,
            segment.sample_rate,
            window_start,
            0.05 * np.ptp(segment.sample),
            color="red",
            label="Annotated points",
        )

    _set_title(
        ax,
        title=title,
        default=f"ECG Signal — {start / signal.sample_rate:.2f}-{(end / signal.sample_rate):.2f} s",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [mV]")
    ax.legend()


@with_axes
def plot_amplitude_distribution(
    signal: ECGSignal,
    physiological_limit_mv: float = 5.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    sns.histplot(
        signal.sample,
        bins="auto",
        kde=True,
        linewidth=0,
        ax=ax,
        label="Amplitude distribution",
        **plot_kwargs,
    )

    ax.axvline(
        -physiological_limit_mv,
        color="red",
        linestyle="--",
        label=f"Min threshold ({-physiological_limit_mv} mV)",
    )
    ax.axvline(
        physiological_limit_mv,
        color="red",
        linestyle="--",
        label=f"Max threshold ({physiological_limit_mv} mV)",
    )

    _set_title(ax, title=title, default="Amplitude distribution")

    ax.set_xlabel("Amplitude [mV]")
    ax.set_ylabel("Count")
    ax.legend()


@with_axes
def plot_average_beat(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
    window_ms: tuple[int, int] = (200, 400),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    left = round(window_ms[0] * signal.sample_rate / 1000)
    right = round(window_ms[1] * signal.sample_rate / 1000)

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

    ax.plot(
        time,
        mean_beat,
        label="Mean beat",
    )

    ax.fill_between(
        time,
        mean_beat - std_beat,
        mean_beat + std_beat,
        alpha=0.25,
        label="±1 SD",
    )

    ax.axvline(
        0,
        color="red",
        linestyle="--",
        label="R-peak",
    )

    _set_title(ax, title=title, default=f"ECG Beat Overlay ({len(beats_arr)} beats)")

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude [mV]")
    ax.legend()


@with_axes
def plot_rr_tachogram(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    rr_intervals = compute_rr_intervals(signal, r_peaks)
    beat_idx = np.arange(1, len(r_peaks.sample))

    ax.plot(
        beat_idx,
        rr_intervals,
        "-o",
        markersize=3,
        label="RR interval",
    )

    _set_title(ax, title=title, default="RR Tachogram")

    ax.set_xlabel("Beat index")
    ax.set_ylabel("RR interval [ms]")
    ax.legend()


@with_axes
def plot_rr_distribution(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    rr_intervals = compute_rr_intervals(signal, r_peaks)

    sns.histplot(
        rr_intervals,
        bins="auto",
        kde=True,
        linewidth=0,
        ax=ax,
        label="RR interval distribution",
    )

    _set_title(ax, title=title, default="RR Distribution")

    ax.set_xlabel("RR Interval [ms]")
    ax.set_ylabel("Count")
    ax.legend()


@with_axes
def plot_poincare(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    rr_intervals = compute_rr_intervals(signal, r_peaks)

    ax.scatter(
        rr_intervals[:-1],
        rr_intervals[1:],
        s=30,
        label="RRₙ vs RRₙ₊₁",
    )

    min_rr, max_rr = np.min(rr_intervals), np.max(rr_intervals)
    ax.plot(
        [min_rr, max_rr],
        [min_rr, max_rr],
        "r--",
    )

    _set_title(ax, title=title, default="Poincaré Plot")

    ax.set_xlabel("RRₙ [ms]")
    ax.set_ylabel("RRₙ₊₁ [ms]")
    ax.legend()

    ax.set_aspect("auto")


@with_axes
def plot_spectrogram(
    signal: ECGSignal,
    nperseg: int = 256,
    noverlap: int = 200,
    scaling: str = "density",
    mode: str = "magnitude",
    start_sec: float = 0,
    interval_sec: float = 5,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    segment, _, start, end = _get_segment(
        signal,
        start_sec,
        interval_sec,
    )

    freqs, time, S = compute_spectrogram(
        segment,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling=scaling,
        mode=mode,
    )

    S_db = 10 * np.log10(np.maximum(S, 1e-12))

    window_start = start / segment.sample_rate

    time = time + window_start

    fig = ax.figure

    im = ax.pcolormesh(
        time,
        freqs,
        S_db,
        shading="auto",
        cmap="cividis",
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Power [dB]",
    )

    _set_title(
        ax,
        title=title,
        default=f"Spectrogram — {start / signal.sample_rate:.2f}-{(end / signal.sample_rate):.2f} s",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")


@with_axes
def plot_wavelet_scalogram(
    signal: ECGSignal,
    wavelet: str = "morl",
    start_sec: float = 0,
    interval_sec: float = 5,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    segment, time, start, end = _get_segment(
        signal,
        start_sec,
        interval_sec,
    )

    coef, freqs = compute_wavelet(
        segment,
        wavelet=wavelet,
    )

    power = np.abs(coef)

    fig = ax.figure

    im = ax.imshow(
        power,
        extent=[time[0], time[-1], freqs[-1], freqs[0]],
        aspect="auto",
        cmap="cividis",
        origin="upper",
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Magnitude",
    )

    _set_title(
        ax,
        title=title,
        default=f"Wavelet Scalogram ({wavelet}) — {start / signal.sample_rate:.2f}-{(end / signal.sample_rate):.2f} s",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency  [Hz]")


@with_axes
def plot_baseline_wander(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
    start_sec: float = 0,
    interval_sec: float = 5,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    segment, time, start, end = _get_segment(
        signal,
        start_sec,
        interval_sec,
    )

    baseline = compute_baseline(
        segment,
        window1_ms=window1_ms,
        window2_ms=window2_ms,
    )

    ax.plot(
        time,
        baseline,
        color="red",
        label="Estimated Baseline",
        **plot_kwargs,
    )

    _set_title(
        ax,
        title=title,
        default=f"Baseline Wander — {start / signal.sample_rate:.2f}-{(end / signal.sample_rate):.2f} s",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [mV]")
    ax.legend()


@with_axes
def plot_fft(
    signal: ECGSignal,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    freqs, magnitude = compute_fft(signal)

    ax.plot(freqs, magnitude, **plot_kwargs)

    _set_title(
        ax,
        title=title,
        default="FFT Magnitude Spectrum",
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [mV]")


@with_axes
def plot_psd(
    signal: ECGSignal,
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    freqs, psd = compute_psd(
        signal,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )

    ax.semilogy(freqs, psd, **plot_kwargs)

    _set_title(
        ax,
        title=title,
        default="Power Spectral Density",
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [mV²/Hz]")


@with_axes
def plot_detection_results(
    signal: ECGSignal,
    predicted: ECGAnnotation,
    start_sec: float = 0,
    interval_sec: float = 5,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    segment, time, start, end = _get_segment(
        signal,
        start_sec,
        interval_sec,
    )

    window_start = start / signal.sample_rate

    ax.plot(time, segment.sample, label="ECG Signal")

    _plot_annotations(
        ax,
        segment.sample,
        segment.annotation,
        segment.sample_rate,
        window_start,
        0.05 * np.ptp(segment.sample),
        marker="v",
        color="red",
        label="Ground truth",
    )

    predicted_segment = predicted.segment(start, end)

    _plot_annotations(
        ax,
        segment.sample,
        predicted_segment,
        segment.sample_rate,
        window_start,
        -0.1 * np.ptp(segment.sample),
        marker="^",
        color="green",
        label="Prediction",
    )

    _set_title(
        ax,
        title=title,
        default=f"ECG Signal Detection Result — {start / signal.sample_rate:.2f}-{end / signal.sample_rate:.2f} s",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [mV]")
    ax.legend()


WAVE_BOUNDS = {
    "P": (Fiducials.P_ONSET, Fiducials.P_OFFSET),
    "QRS": (Fiducials.QRS_ONSET, Fiducials.QRS_OFFSET),
    "T": (Fiducials.T_ONSET, Fiducials.T_OFFSET),
}


def highlight_wave_bounds(
    signal: ECGSignal,
    wave: str,
    ax: plt.Axes,
    start_sec: float = 0,
    interval_sec: float = 10,
    onset_color: str = "tab:gray",
    offset_color: str = "tab:gray",
    span_color: str = "gray",
) -> None:
    if signal.annotation is None:
        return

    onset_f, offset_f = WAVE_BOUNDS[wave]
    fs = signal.sample_rate
    start, end = compute_window(len(signal), fs, start_sec, interval_sec)

    def _bounds(fiducial):
        s = signal.annotation.filter(fiducial).sample
        return s[(s >= start) & (s < end)]

    onsets, offsets = _bounds(onset_f), _bounds(offset_f)

    for on in onsets:
        i = np.searchsorted(offsets, on, side="right")
        if i < len(offsets):
            ax.axvspan(on / fs, offsets[i] / fs, color=span_color, alpha=0.1, zorder=1)

    for samples, color, label, marker in (
        (onsets, onset_color, f"{wave} onset", ">"),
        (offsets, offset_color, f"{wave} offset", "<"),
    ):
        if len(samples) == 0:
            continue
        x = samples / fs
        for xi in x:
            ax.axvline(xi, color=color, linestyle="--", linewidth=1, zorder=2)
        ax.scatter(
            x,
            signal.sample[samples],
            s=40,
            color=color,
            marker=marker,
            zorder=4,
            label=label,
        )

    ax.legend(loc="upper right")
