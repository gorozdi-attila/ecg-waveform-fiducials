from scipy.signal import butter, iirnotch, tf2sos, sosfiltfilt

from ecg_waveform.core import ECGSignal

from ecg_waveform.utils import _compute_baseline


def remove_baseline(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
) -> ECGSignal:
    baseline = _compute_baseline(
        signal,
        window1_ms=window1_ms,
        window2_ms=window2_ms,
    )

    return signal.with_sample(signal.sample - baseline)


def notch_filter(
    signal: ECGSignal,
    freq: float = 50.0,
    quality_factor: float = 30.0,
) -> ECGSignal:
    b, a = iirnotch(freq, quality_factor, signal.sample_rate)
    sos = tf2sos(b, a)

    return signal.with_sample(sosfiltfilt(sos, signal.sample))


def butter_filter(
    signal: ECGSignal,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 4,
) -> ECGSignal:

    if low_cutoff is not None and high_cutoff is not None:
        btype = "band"
        wn = [low_cutoff, high_cutoff]
    elif low_cutoff is not None:
        btype = "high"
        wn = low_cutoff
    elif high_cutoff is not None:
        btype = "low"
        wn = high_cutoff
    else:
        raise ValueError("Specify low_cutoff and/or high_cutoff.")

    sos = butter(
        order,
        wn,
        btype=btype,
        fs=signal.sample_rate,
        output="sos",
    )

    return signal.with_sample(sosfiltfilt(sos, signal.sample, padtype=None))
