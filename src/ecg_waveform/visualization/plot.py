from collections.abc import Callable
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecg_waveform.core import ECGAnnotation, ECGSignal
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
    """
    Get the Matplotlib figure and axes for a plot.

    If no axes are provided, a new figure and axes are created.
    Otherwise, the provided axes and their parent figure are returned.

    Parameters
    ----------
    ax : plt.Axes or None
        Existing Matplotlib axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    fig : plt.Figure
        The Matplotlib figure containing the axes.
    ax : plt.Axes
        The axes to draw on.
    created : bool
        Whether a new figure and axes were created.
    """

    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax, True

    return ax.figure, ax, False


def _set_title(ax: plt.Axes, title: str | None, default: str) -> None:
    """
    Set the title of a Matplotlib axes.

    The provided title is used when available; otherwise, the default title is applied.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes on which to set the title.
    title : str or None
        Custom plot title. If None, the default title is used.
    default : str
        Default plot title used when no custom title is provided.
    """
    ax.set_title(title if title is not None else default)


def _get_segment(
    signal: ECGSignal,
    start_sec: float,
    interval_sec: float,
) -> tuple[ECGSignal, np.ndarray, int, int]:
    """
    Extract a segment of an ECG signal over a specified time interval.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal from which to extract the segment.
    start_sec : float
        Start time of the segment in seconds.
    interval_sec : float
        Duration of the segment in seconds.

    Returns
    -------
    segment : ECGSignal
        Extracted ECG signal segment.
    time : np.ndarray
        Time values corresponding to the samples in the segment, in seconds.
    start : int
        Start index of the segment in the original signal.
    end : int
        End index of the segment in the original signal.
    """

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
    """
    Decorate a plotting function with Matplotlib axes management and styling.

    The decorator provides an optional ``ax`` keyword argument to the wrapped function.
    If no axes are supplied, a new figure and axes are created.
    Major and minor grid lines are configured automatically, and the figure is tightened when newly created.

    Parameters
    ----------
    plot_fn : Callable
        Plotting function to decorate. The function must accept an ax keyword argument.

    Returns
    -------
    Callable
        Decorated plotting function that accepts an optional ax argument and returns the Matplotlib figure and axes.
    """

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
    """
    Plot ECG annotations and their symbols on a Matplotlib axes.

    Annotations are plotted at their corresponding sample positions and labeled with their annotation symbols.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes on which to plot the annotations.
    samples : np.ndarray
        ECG samples used to determine the y-coordinate of each annotation.
    annotation : ECGAnnotation
        ECG annotations containing sample indices and annotation symbols.
    sample_rate : float
        Sampling frequency of the ECG signal in samples per second.
    window_start : float
        Start time of the plotted window in seconds.
    offset : float
        Vertical offset applied to annotation symbols.
    color : str
        Color used for annotation markers and symbols.
    label : str
        Label assigned to the annotation markers in the plot legend.
    marker : str, optional
        Matplotlib marker style used for the annotation points. Default is ``"o"``.
    """

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
    """
    Plot an ECG signal over a specified time interval.

    The selected signal segment is plotted together with its annotations, if enabled.
    If no axes are provided, a new figure and axes are created.
    The plot can be customized using additional keyword arguments passed directly to Matplotlib's plot function.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal to plot.
    start_sec : float, optional
        Start time of the plotted interval in seconds.
        Default is 0.
    interval_sec : float, optional
        Duration of the plotted interval in seconds.
        Default is 5.
    show_annotation : bool, optional
        Whether to display the ECG annotations.
        Default is True.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the signal. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, a title containing the plotted time interval is generated automatically.
    **plot_kwargs
        Additional keyword arguments passed to Matplotlib's plot function.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the plot.
    ax : plt.Axes
        Matplotlib axes containing the ECG signal.
    """

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
    """
    Plot the amplitude distribution of an ECG signal.

    The distribution is shown as a histogram with a kernel density estimate.
    Vertical lines indicate the positive and negative physiological amplitude limits.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal whose sample amplitudes are plotted.
    physiological_limit_mv : float, optional
        Absolute physiological amplitude limit in millivolts. Vertical
        threshold lines are plotted at both positive and negative values.
        Default is 5.0 mV.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the distribution. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, Amplitude distribution is used.
    **plot_kwargs
        Additional keyword arguments passed to Seaborn's histplot function.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the plot.
    ax : plt.Axes
        Matplotlib axes containing the amplitude distribution.
    """
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
    """
    Plot the mean ECG beat and its variability around R-peaks.

    Beats are extracted around each R-peak using the specified pre- and post-peak time windows.
    The plot shows the mean beat with a shaded region corresponding to plus or minus one standard deviation across the extracted beats.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal from which the beats are extracted.
    r_peaks : ECGAnnotation
        ECG annotations containing the sample indices of R-peaks.
    window_ms : tuple[int, int], optional
        Time window around each R-peak in milliseconds, specified as (before, after).
        Default is (200, 400).
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the plot. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, a title containing the number of valid beats is generated automatically.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the plot.
    ax : plt.Axes
        Matplotlib axes containing the beat overlay.

    Raises
    ------
    ValueError
        If no beats fall completely within the specified window.
    """

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
    """
    Plot an RR tachogram from detected R-peaks.

    The RR intervals are plotted against their corresponding beat indices to visualize the variation in time between consecutive R-peaks.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal used to compute the RR intervals.
    r_peaks : ECGAnnotation
        ECG annotations containing the sample indices of detected R-peaks.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the tachogram. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, RR Tachogram is used.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the tachogram.
    ax : plt.Axes
        Matplotlib axes containing the RR intervals.
    """
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
    """
    Plot the distribution of RR intervals.

    The RR intervals are visualized as a histogram with a kernel density estimate to show their distribution.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal used to compute the RR intervals.
    r_peaks : ECGAnnotation
        ECG annotations containing the sample indices of detected R-peaks.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the distribution. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, RR Distribution is used.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the RR interval distribution.
    ax : plt.Axes
        Matplotlib axes containing the distribution.
    """

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
    """
    Plot a Poincaré plot of consecutive RR intervals.

    Each point represents a pair of consecutive RR intervals, with thecurrent interval plotted against the subsequent interval.
    A dashed diagonal line indicates equal consecutive RR intervals.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal used to compute the RR intervals.
    r_peaks : ECGAnnotation
        ECG annotations containing the sample indices of detected R-peaks.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the plot. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, Poincaré Plot is used.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the Poincaré plot.
    ax : plt.Axes
        Matplotlib axes containing the consecutive RR interval pairs.
    """

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
    """
    Plot the time-frequency representation of an ECG signal.

    The selected signal segment is transformed into a spectrogram and displayed as a time-frequency heatmap.
    Spectral values are converted to decibels before plotting.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the spectrogram.
    nperseg : int, optional
        Number of samples per segment used for the spectral analysis.
        Default is 256.
    noverlap : int, optional
        Number of samples overlapping between consecutive segments.
        Default is 200.
    scaling : str, optional
        Scaling applied to the spectrogram. Passed to the underlying spectrogram computation.
        Default is density.
    mode : str, optional
        Spectrogram output mode. Passed to the underlying spectrogram computation.
        Default is magnitude.
    start_sec : float, optional
        Start time of the plotted interval in seconds.
        Default is 0.
    interval_sec : float, optional
        Duration of the plotted interval in seconds.
        Default is 5.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the spectrogram. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, a title containing the plotted time interval is generated automatically.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the spectrogram.
    ax : plt.Axes
        Matplotlib axes containing the spectrogram.
    """

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
    """
    Plot a wavelet scalogram of an ECG signal.

    The selected signal segment is transformed using the continuous wavelet transform,
    and the magnitude of the resulting wavelet coefficients is displayed as a time-frequency heatmap.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the wavelet scalogram.
    wavelet : str, optional
        Name of the wavelet used for the continuous wavelet transform.
        Default is morl.
    start_sec : float, optional
        Start time of the plotted interval in seconds.
        Default is 0.
    interval_sec : float, optional
        Duration of the plotted interval in seconds.
        Default is 5.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the scalogram. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, a title containing the wavelet name and plotted time interval is generated automatically.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the scalogram.
    ax : plt.Axes
        Matplotlib axes containing the wavelet scalogram.
    """

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
    """
    Plot the estimated baseline wander of an ECG signal.

    The selected signal segment is used to estimate the baseline with the specified smoothing windows.
    The resulting baseline is plotted as a function of time.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal from which to estimate the baseline.
    window1_ms : int, optional
        Width of the first baseline estimation window in milliseconds.
        Default is 200 ms.
    window2_ms : int, optional
        Width of the second baseline estimation window in milliseconds.
        Default is 600 ms.
    start_sec : float, optional
        Start time of the plotted interval in seconds.
        Default is 0.
    interval_sec : float, optional
        Duration of the plotted interval in seconds.
        Default is 5.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the baseline. If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, a title containing the plotted time interval is generated automatically.
    **plot_kwargs
        Additional keyword arguments passed to Matplotlib's ``plot`` function.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the baseline estimate.
    ax : plt.Axes
        Matplotlib axes containing the baseline estimate.
    """

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
    """
    Plot the magnitude spectrum of an ECG signal using the Fast Fourier Transform.

    The Fourier transform is computed for the complete signal and the
    resulting magnitude spectrum is plotted as a function of frequency.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the Fourier transform.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the spectrum. If None,
        new figure and axes are created.
    title : str or None, optional
        Plot title. If None, ``"FFT Magnitude Spectrum"`` is used.
    **plot_kwargs
        Additional keyword arguments passed to Matplotlib's ``plot`` function.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the FFT magnitude spectrum.
    ax : plt.Axes
        Matplotlib axes containing the FFT magnitude spectrum.
    """

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
    """
    Plot the power spectral density of an ECG signal.

    The power spectral density is computed using Welch's method and plotted
    on a logarithmic y-axis.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the power spectral density.
    nperseg : int or None, optional
        Number of samples in each segment used for the spectral estimation.
        If None, the default value of the underlying PSD computation is used.
    window : str, optional
        Window function applied to each segment. Default is ``"hann"``.
    noverlap : int or None, optional
        Number of samples that overlap between consecutive segments. If None,
        the default value of the underlying PSD computation is used.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the PSD. If None,
        new figure and axes are created.
    title : str or None, optional
        Plot title. If None, ``"Power Spectral Density"`` is used.
    **plot_kwargs
        Additional keyword arguments passed to Matplotlib's
        ``semilogy`` function.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the power spectral density.
    ax : plt.Axes
        Matplotlib axes containing the power spectral density.
    """

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
    """
    Plot ECG detection results against the ground-truth annotations.

    The selected ECG segment is plotted together with the ground-truth
    annotations and predicted annotations. Different markers and colors
    are used to distinguish the two annotation sets.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal containing the ground-truth annotations.
    predicted : ECGAnnotation
        Predicted ECG annotations to compare with the ground truth.
    start_sec : float, optional
        Start time of the plotted interval in seconds. Default is 0.
    interval_sec : float, optional
        Duration of the plotted interval in seconds. Default is 5.
    ax : plt.Axes or None, optional
        Existing Matplotlib axes on which to draw the detection results.
        If None, new figure and axes are created.
    title : str or None, optional
        Plot title. If None, a title containing the plotted time interval
        is generated automatically.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure containing the detection results.
    ax : plt.Axes
        Matplotlib axes containing the ECG signal and annotations.
    """

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
