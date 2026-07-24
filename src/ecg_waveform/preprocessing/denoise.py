import numpy as np

from scipy.signal import savgol_filter, medfilt

import pywt

from ecg_waveform.core import ECGSignal


def wavelet_denoise(
    signal: ECGSignal,
    wavelet: str = "db4",
    level: int = 4,
) -> ECGSignal:
    coeffs = pywt.wavedec(signal.sample, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    denoised = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]

    return signal.with_sample(pywt.waverec(denoised, wavelet)[: len(signal)])


def savgol_smooth(
    signal: ECGSignal,
    window_ms: float = 50.0,
    polyorder: int = 3,
) -> ECGSignal:
    window = int((window_ms / 1000) * signal.sample_rate)
    if window % 2 == 0:
        window += 1

    return signal.with_sample(savgol_filter(signal.sample, window, polyorder))


def median_smooth(
    signal: ECGSignal,
    kernel_ms: float = 20.0,
) -> ECGSignal:
    kernel = int((kernel_ms / 1000) * signal.sample_rate)
    if kernel % 2 == 0:
        kernel += 1
    return signal.with_sample(medfilt(signal.sample, kernel))
