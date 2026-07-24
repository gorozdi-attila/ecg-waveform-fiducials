import numpy as np

from scipy.stats import kurtosis

from ecg_waveform.core import ECGSignal

from ecg_waveform.utils import compute_baseline, compute_psd


def kurtosis_sqi(
    signal: ECGSignal,
    fisher: bool = True,
) -> float:
    x = np.asarray(signal.sample, dtype=float)

    if x.size == 0 or np.std(x) == 0:
        return float("nan")

    return float(kurtosis(x, fisher=fisher, bias=True))


def band_power_ratio(
    signal: ECGSignal,
    band: tuple[float, float | None],
    reference_band: tuple[float, float | None] = (0.0, None),
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> float:
    freqs, psd = compute_psd(
        signal,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )

    if freqs.size == 0:
        return float("nan")

    def _band_power(low: float, high: float | None) -> float:
        high = freqs[-1] if high is None else high
        mask = (freqs >= low) & (freqs <= high)

        if not np.any(mask):
            return 0.0

        return float(np.trapezoid(psd[mask], freqs[mask]))

    if _band_power(*reference_band) == 0.0:
        return float("nan")

    return _band_power(*band) / _band_power(*reference_band)


def qrs_power_sqi(
    signal: ECGSignal,
    qrs_band: tuple[float, float] = (5.0, 15.0),
    reference_band: tuple[float, float] = (0.5, 40.0),
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
) -> float:
    return band_power_ratio(
        signal,
        band=qrs_band,
        reference_band=reference_band,
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )


def powerline_noise_ratio(
    signal: ECGSignal,
    powerline_freq: float = 50.0,
    bandwidth: float = 1.0,
    nperseg: int | None = None,
    window: str = "hann",
    noverlap: int | None = None,
):
    band = (powerline_freq - bandwidth, powerline_freq + bandwidth)

    return band_power_ratio(
        signal,
        band=band,
        reference_band=(0.0, None),
        nperseg=nperseg,
        window=window,
        noverlap=noverlap,
    )


def baseline_wander_ratio(
    signal: ECGSignal,
    window1_ms: int = 200,
    window2_ms: int = 600,
) -> float:
    x = np.asarray(signal.sample, dtype=float)

    if x.size == 0:
        return float("nan")

    signal_power = float(np.mean(x**2))
    if signal_power == 0.0:
        return float("nan")

    baseline = compute_baseline(signal, window1_ms=window1_ms, window2_ms=window2_ms)
    baseline_power = float(np.mean(np.asarray(baseline, dtype=float) ** 2))

    return baseline_power / signal_power


def flatline_ratio(
    signal: ECGSignal, slope_threshold: float = 1e-4, min_run_ms: float = 500.0
) -> float:
    x = np.asarray(signal.sample, dtype=float)

    if x.size < 2:
        return 0.0

    min_run_samples = max(1, int(round(min_run_ms / 1000.0 * signal.sample_rate)))

    dt = 1.0 / signal.sample_rate
    slope = np.abs(np.diff(x, prepend=x[0])) / dt
    is_flat = slope < slope_threshold

    change_points = np.flatnonzero(np.diff(is_flat.astype(np.int8))) + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [is_flat.size]))
    run_is_flat = is_flat[run_starts]
    run_lengths = run_ends - run_starts

    qualifying = run_is_flat & (run_lengths >= min_run_samples)
    flat_samples = int(run_lengths[qualifying].sum())

    return flat_samples / x.size


def beat_agreement_sqi(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    sample_rate: int,
    tolerance_ms: float = 50.0,
) -> float:
    peaks_a = np.asarray(peaks_a, dtype=np.int64)
    peaks_b = np.asarray(peaks_b, dtype=np.int64)

    if peaks_a.size == 0 and peaks_b.size == 0:
        return float("nan")
    if peaks_a.size == 0 or peaks_b.size == 0:
        return 0.0

    tolerance_samples = tolerance_ms / 1000.0 * sample_rate

    i, j, n_matched = 0, 0, 0
    while i < peaks_a.size and j < peaks_b.size:
        diff = peaks_a[i] - peaks_b[j]
        if abs(diff) <= tolerance_samples:
            n_matched += 1
            i += 1
            j += 1
        elif diff > 0:
            j += 1
        else:
            i += 1

    return 2 * n_matched / (peaks_a.size + peaks_b.size)


def rr_plausibility_ratio(
    peaks: np.ndarray,
    sample_rate: int,
    min_bpm: float = 30.0,
    max_bpm: float = 220.0,
) -> float:
    peaks = np.asarray(peaks, dtype=np.int64)

    if peaks.size < 2:
        return float("nan")

    rr_ms = np.diff(peaks) / sample_rate * 1000.0
    instantaneous_bpm = 60_000.0 / rr_ms

    plausible = (instantaneous_bpm >= min_bpm) & (instantaneous_bpm <= max_bpm)

    return float(np.mean(plausible))
