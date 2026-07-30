from scipy.signal import (
    bessel,
    butter,
    cheby1,
    cheby2,
    ellip,
    iirnotch,
    sosfiltfilt,
    tf2sos,
)

from ecg_waveform.core import ECGSignal
from ecg_waveform.utils import compute_baseline


def _iir_filter(
    design_fn,
    signal,
    low_cutoff=None,
    high_cutoff=None,
    order=4,
    *design_args,
    **kwargs,
) -> ECGSignal:
    if low_cutoff is not None and high_cutoff is not None:
        btype = "bandpass"
        wn = [low_cutoff, high_cutoff]
    elif low_cutoff is not None:
        btype = "highpass"
        wn = low_cutoff
    elif high_cutoff is not None:
        btype = "lowpass"
        wn = high_cutoff
    else:
        raise ValueError("Specify low_cutoff and/or high_cutoff.")

    sos = design_fn(
        order,
        *design_args,
        wn,
        btype=btype,
        fs=signal.sample_rate,
        output="sos",
        **kwargs,
    )

    return signal.with_sample(sosfiltfilt(sos, signal.sample, padtype=None))


def median_baseline_filter(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
) -> ECGSignal:
    baseline = compute_baseline(
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
    filtered = sosfiltfilt(sos, signal.sample)

    return signal.with_sample(filtered)


def butterworth_filter(
    signal: ECGSignal,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 4,
) -> ECGSignal:
    return _iir_filter(
        butter,
        signal,
        low_cutoff,
        high_cutoff,
        order,
    )


def chebyshev1_filter(
    signal: ECGSignal,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 4,
    ripple_db: float = 1.0,
) -> ECGSignal:
    return _iir_filter(
        cheby1,
        signal,
        low_cutoff,
        high_cutoff,
        order,
        ripple_db, 
    )


def chebyshev2_filter(
    signal: ECGSignal,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 4,
    attenuation_db: float = 40.0,
) -> ECGSignal:
    return _iir_filter(
        cheby2,
        signal,
        low_cutoff,
        high_cutoff,
        order,
        attenuation_db,
    )


def elliptic_filter(
    signal: ECGSignal,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 4,
    ripple_db: float = 1.0,
    attenuation_db: float = 40.0,
) -> ECGSignal:
    return _iir_filter(
        ellip,
        signal,
        low_cutoff,
        high_cutoff,
        order,
        ripple_db,
        attenuation_db,
    )


def bessel_filter(
    signal: ECGSignal,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 4,
) -> ECGSignal:
    return _iir_filter(
        bessel,
        signal,
        low_cutoff,
        high_cutoff,
        order,
    )
