import numpy as np
import pywt
from scipy.ndimage import median_filter
from scipy.signal import spectrogram, welch

from ecg_waveform.core import ECGAnnotation, ECGSignal


def compute_window(
    n_sample: int,
    sample_rate: int,
    start_sec: float,
    interval_sec: float,
) -> tuple[int, int]:
    """
    Compute sample indices for a time-based signal window.

    The requested time interval is converted from seconds to sample indices and clipped to the available signal length.

    Parameters
    ----------
    n_sample : int
        Total number of samples in the signal.
    sample_rate : int
        Sampling frequency of the signal in samples per second.
    start_sec : float
        Start time of the requested window in seconds.
    interval_sec : float
        Duration of the requested window in seconds.

    Returns
    -------
    start : int
        Start sample index of the window.
    end : int
        End sample index of the window.

    Raises
    ------
    ValueError
        If n_sample or sample_rate is not positive.
    ValueError
        If start_sec is negative or interval_sec is not positive.
    ValueError
        If the requested window is outside the signal bounds or has zero length.
    """

    if n_sample <= 0:
        raise ValueError("n_sample must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if start_sec < 0:
        raise ValueError("start_sec must be non-negative.")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive.")

    start = round(start_sec * sample_rate)
    end = round((start_sec + interval_sec) * sample_rate)

    start = min(start, n_sample)
    end = min(end, n_sample)

    if start >= end:
        raise ValueError(
            "Requested window is outside the signal bounds or has zero length."
        )

    return start, end


def compute_rr_intervals(
    signal: ECGSignal,
    r_peaks: ECGAnnotation,
) -> np.ndarray:
    """
    Compute RR intervals from consecutive R-peaks.

    The difference between consecutive R-peak sample indices is converted from samples to milliseconds using the signal sampling frequency.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal associated with the detected R-peaks.
    r_peaks : ECGAnnotation
        ECG annotations containing the sample indices of detected R-peaks.

    Returns
    -------
    np.ndarray
        RR intervals between consecutive R-peaks, in milliseconds.

    Raises
    ------
    ValueError
        If fewer than two R-peaks are provided.
    """

    if len(r_peaks.sample) < 2:
        raise ValueError("At least two R-peaks are required.")

    return np.diff(r_peaks.sample) / signal.sample_rate * 1000


def compute_baseline(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
) -> np.ndarray:
    """
    Estimate the baseline wander of an ECG signal.

    The baseline is estimated using two consecutive median filters with configurable window sizes.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to estimate the baseline.
    window1_ms : int, optional
        Width of the first median filter window in milliseconds.
        Default is 200 ms.
    window2_ms : int, optional
        Width of the second median filter window in milliseconds.
        Default is 600 ms.

    Returns
    -------
    np.ndarray
        Estimated baseline signal with the same length as the input signal.
    """

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
    """
    Compute the power spectral density of an ECG signal using Welch's method.

    If nperseg is not specified, its value is selected automatically based on the length of the input signal.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the power spectral density.
    nperseg : int or None, optional
        Number of samples in each segment used for spectral estimation.
        If None, the value is selected automatically based on the signal length.
    window : str, optional
        Window function applied to each segment.
        Default is hann.
    noverlap : int or None, optional
        Number of samples that overlap between consecutive segments.
        If None, the default value of Welch's method is used.

    Returns
    -------
    freqs : np.ndarray
        Frequencies corresponding to the estimated power spectral density,
        in hertz.
    psd : np.ndarray
        Estimated power spectral density values.

    Raises
    ------
    ValueError
        If nperseg is not positive.
    ValueError
        If noverlap is negative or greater than or equal to nperseg
    """

    if len(signal.sample) == 0:
        return np.array([]), np.array([])

    if nperseg is None:
        nperseg = min(len(signal.sample), 4096, max(256, len(signal.sample) // 4))

    if nperseg <= 0:
        raise ValueError("nperseg must be positive.")

    if noverlap is not None:
        if noverlap < 0:
            raise ValueError("noverlap must be non-negative.")
        if noverlap >= nperseg:
            raise ValueError("noverlap must be smaller than nperseg.")

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
    """
    Compute the one-sided magnitude spectrum of an ECG signal.

    The signal is mean-centered and multiplied by a Hann window before computing the real-valued Fast Fourier Transform.
    The resulting magnitude spectrum is normalized by the sum of the window coefficients.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the Fourier transform.

    Returns
    -------
    freqs : np.ndarray
        Frequencies corresponding to the magnitude spectrum, in hertz.
    magnitude : np.ndarray
        One-sided magnitude spectrum of the ECG signal, in millivolts.
    """
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
    """
    Compute the time-frequency representation of an ECG signal.

    The signal is divided into overlapping segments and transformed using a short-time Fourier transform to obtain its spectrogram.

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
        Scaling applied to the spectrogram. Passed to scipy.signal.spectrogram.
        Default is density.
    mode : str, optional
        Spectrogram output mode. Passed to scipy.signal.spectrogram.
        Default is magnitude.

    Returns
    -------
    freqs : np.ndarray
        Sample frequencies of the spectrogram, in hertz.
    time : np.ndarray
        Segment times corresponding to the spectrogram columns, in seconds.
    S : np.ndarray
        Spectrogram values.

    Raises
    ------
    ValueError
        If nperseg is not positive.
    ValueError
        If noverlap is negative or greater than or equal to nperseg.
    """
    if nperseg <= 0:
        raise ValueError("nperseg must be positive.")

    if noverlap < 0:
        raise ValueError("noverlap must be non-negative.")

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
    """
    Compute the continuous wavelet transform of an ECG signal.

    The continuous wavelet transform is computed using PyWavelets.
    If no scales are provided, scales from 1 to 127 are used.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the wavelet transform.
    wavelet : str, optional
        Name of the wavelet used for the continuous wavelet transform.
        Default is morl.
    scales : np.ndarray or None, optional
        Scales used for the wavelet transform. If None, scales from 1 to 127 are used.

    Returns
    -------
    coef : np.ndarray
        Wavelet coefficients for each scale and signal sample.
    freqs : np.ndarray
        Frequencies corresponding to the wavelet scales, in hertz.
    """
    if scales is None:
        scales = np.arange(1, 128)

    coef, freqs = pywt.cwt(
        signal.sample,
        scales,
        wavelet,
        sampling_period=1 / signal.sample_rate,
    )

    return coef, freqs


def compute_snr(
    signal: ECGSignal,
    reference: ECGSignal,
) -> float:
    """
    Compute the signal-to-noise ratio (SNR) of an ECG signal relative to a reference signal.

    The noise is defined as the difference between the input signal and the
    reference signal. SNR is computed from the mean-square power of the
    reference signal and the noise signal.

    Parameters
    ----------
    signal : ECGSignal
        ECG signal for which to compute the SNR.
    reference : ECGSignal
        Reference ECG signal representing the underlying clean signal.

    Returns
    -------
    float
        Signal-to-noise ratio in decibels (dB).

    Raises
    ------
    ValueError
        If the signals have different lengths.
    ValueError
        If the signals have different sampling rates.
    ValueError
        If the signal power is zero.
    """
    if len(signal.sample) != len(reference.sample):
        raise ValueError("signal and reference must have the same length.")

    if signal.sample_rate != reference.sample_rate:
        raise ValueError("signal and reference must have the same sample rate.")

    x = signal.sample.astype(float)
    ref = reference.sample.astype(float)

    noise = x - ref

    signal_power = np.mean(x**2)
    noise_power = np.mean(noise**2)

    if signal_power == 0:
        raise ValueError("Reference signal has zero power.")

    if noise_power == 0:
        return np.inf

    return 10 * np.log10(signal_power / noise_power)
