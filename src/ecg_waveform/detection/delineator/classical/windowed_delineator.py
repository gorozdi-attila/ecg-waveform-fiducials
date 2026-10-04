from dataclasses import dataclass

import numpy as np

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials
from ecg_waveform.signal.preprocessing import butterworth_filter

from ..base_delineator import BaseDelineator
from ..delineator_registry import register_delineator
from ...common import first_below_persistent, moving_average


@dataclass(slots=True)
class _WaveResult:
    onset: int | None
    peak: int | None
    offset: int | None
    amplitude: float = 0.0
    noise: float = 0.0


@register_delineator("windowed")
@dataclass(slots=True)
class WindowedDelineator(BaseDelineator):
    qrs_low_cutoff: float = 5.0
    qrs_high_cutoff: float = 18.0
    qrs_order: int = 4
    qrs_search_window_s: float = 0.12
    qrs_slope_threshold_ratio: float = 0.08
    qrs_slope_smoothing_s: float = 0.008

    p_low_cutoff: float = 0.5
    p_high_cutoff: float = 10.0
    p_order: int = 2
    p_search_before_s: float = 0.30
    p_search_gap_s: float = 0.02
    p_min_gap_from_prev_beat_s: float = 0.15
    p_amplitude_threshold_ratio: float = 0.08
    p_noise_multiplier: float = 2.0
    p_boundary_persistence_s: float = 0.006

    t_low_cutoff: float = 0.5
    t_high_cutoff: float = 8.0
    t_order: int = 2
    t_search_after_s: float = 0.50
    t_search_gap_s: float = 0.02
    t_min_rr_fraction: float = 0.50
    t_qrs_guard_s: float = 0.04
    t_amplitude_threshold_ratio: float = 0.08
    t_noise_multiplier: float = 2.0
    t_boundary_persistence_s: float = 0.008

    minimum_wave_samples: int = 5
    boundary_smoothing_s: float = 0.008

    @property
    def supported_points(self) -> frozenset[Fiducials]:
        return frozenset(
            {
                Fiducials.P_ONSET,
                Fiducials.P_PEAK,
                Fiducials.P_OFFSET,
                Fiducials.QRS_ONSET,
                Fiducials.Q_PEAK,
                Fiducials.S_PEAK,
                Fiducials.QRS_OFFSET,
                Fiducials.T_ONSET,
                Fiducials.T_PEAK,
                Fiducials.T_OFFSET,
            }
        )

    

    def _delineate_qrs_samples(
        self,
        signal: ECGSignal,
        current_r: int,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        n = len(signal)

        window = max(1, int(round(self.qrs_search_window_s * signal.sample_rate)))

        left = max(0, current_r - window)
        right = min(n, current_r + window + 1)

        if right - left < self.minimum_wave_samples:
            return None, None, None, None

        segment = signal.segment(left, right).sample

        r_local = current_r - left

        if segment[r_local] >= 0:
            extrema = np.argmin
        else:
            extrema = np.argmax

        pre = segment[: r_local + 1]
        post = segment[r_local:]

        q_local = int(extrema(pre))
        s_local = int(extrema(post)) + r_local

        q_peak = left + q_local
        s_peak = left + s_local

        derivative = np.abs(np.gradient(segment))

        smoothing = max(1, int(round(self.qrs_slope_smoothing_s * signal.sample_rate)))

        derivative = moving_average(derivative, smoothing)

        max_slope = float(np.max(derivative))

        if max_slope <= 0:
            return None, q_peak, s_peak, None

        threshold = self.qrs_slope_threshold_ratio * max_slope

        persistence = max(1, int(round(0.004 * signal.sample_rate)))

        onset_local = first_below_persistent(
            derivative,
            start=q_local,
            stop=0,
            step=-1,
            threshold=threshold,
            persistence=persistence,
        )

        offset_local = first_below_persistent(
            derivative,
            start=s_local,
            stop=len(derivative) - 1,
            step=1,
            threshold=threshold,
            persistence=persistence,
        )

        qrs_onset = left + onset_local if onset_local is not None else None

        qrs_offset = left + offset_local if offset_local is not None else None

        if (
            qrs_onset is not None
            and qrs_offset is not None
            and not qrs_onset < qrs_offset
        ):
            qrs_onset = None
            qrs_offset = None

        return (
            qrs_onset,
            q_peak,
            s_peak,
            qrs_offset,
        )

    def _find_wave(
        self,
        signal: ECGSignal,
        search_start: int,
        search_end: int,
        amplitude_threshold_ratio: float,
        noise_multiplier: float,
        persistence: int,
    ) -> _WaveResult:
        if search_end <= search_start:
            return _WaveResult(
                None,
                None,
                None,
            )

        segment = signal.segment(search_start, search_end).sample

        if len(segment) < self.minimum_wave_samples:
            return _WaveResult(
                None,
                None,
                None,
            )

        baseline = float(np.median(segment))

        centered = segment - baseline

        deviations = np.abs(centered)

        peak_local = int(np.argmax(deviations))

        peak_amplitude = float(deviations[peak_local])

        if peak_amplitude <= 0:
            return _WaveResult(
                None,
                None,
                None,
                amplitude=peak_amplitude,
            )

        diff = np.diff(segment)

        if len(diff) > 0:
            noise = float(np.median(np.abs(diff)) / np.sqrt(2.0))
        else:
            noise = 0.0

        if noise > 0 and peak_amplitude < noise_multiplier * noise:
            return _WaveResult(
                None,
                None,
                None,
                amplitude=peak_amplitude,
                noise=noise,
            )

        smoothing = max(
            1,
            int(round(self.boundary_smoothing_s * signal.sample_rate)),
        )

        envelope = moving_average(deviations, smoothing)

        peak_amplitude = float(deviations[peak_local])

        threshold = amplitude_threshold_ratio * peak_amplitude

        onset_local = first_below_persistent(
            envelope,
            start=peak_local,
            stop=0,
            step=-1,
            threshold=threshold,
            persistence=persistence,
        )

        offset_local = first_below_persistent(
            envelope,
            start=peak_local,
            stop=len(envelope) - 1,
            step=1,
            threshold=threshold,
            persistence=persistence,
        )

        if onset_local is None or offset_local is None:
            return _WaveResult(
                None,
                search_start + peak_local,
                None,
                amplitude=peak_amplitude,
                noise=noise,
            )

        if not (onset_local < peak_local < offset_local):
            return _WaveResult(
                None,
                None,
                None,
                amplitude=peak_amplitude,
                noise=noise,
            )

        onset = search_start + onset_local

        peak = search_start + peak_local

        offset = search_start + offset_local

        return _WaveResult(
            onset=onset,
            peak=peak,
            offset=offset,
            amplitude=peak_amplitude,
            noise=noise,
        )

    def _delineate_p_samples(
        self,
        signal: ECGSignal,
        boundary_end: int,
        prev_r: int,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        window = int(round(self.p_search_before_s * signal.sample_rate))

        gap = int(round(self.p_search_gap_s * signal.sample_rate))

        search_end = max(0, boundary_end - gap)
        search_start = max(0, search_end - window)

        if prev_r is not None:
            min_start = prev_r + int(round(self.p_min_gap_from_prev_beat_s * signal.sample_rate))

            search_start = max(search_start, min_start)

        if search_end - search_start < self.minimum_wave_samples:
            return None, None, None

        result = self._find_wave(
            signal=signal,
            search_start=search_start,
            search_end=search_end,
            amplitude_threshold_ratio=(self.p_amplitude_threshold_ratio),
            noise_multiplier=(self.p_noise_multiplier),
            persistence=max(
                1, int(round(self.p_boundary_persistence_s * signal.sample_rate))
            ),
        )

        return (
            result.onset,
            result.peak,
            result.offset,
        )

    def _delineate_t_samples(
        self,
        signal: ECGSignal,
        boundary_start: int,
        current_r: int,
        next_r: int | None,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        gap = int(round(self.t_search_gap_s * signal.sample_rate))

        max_window = int(round(self.t_search_after_s * signal.sample_rate))

        search_start = boundary_start + gap

        search_end = current_r + max_window

        if next_r is not None:
            rr = next_r - current_r

            rr_limit = current_r + int(round(self.t_min_rr_fraction * rr))

            qrs_guard = int(round(self.t_qrs_guard_s * signal.sample_rate))

            next_qrs_limit = next_r - qrs_guard

            search_end = min(
                search_end,
                rr_limit,
                next_qrs_limit,
            )

        search_start = max(
            0,
            search_start,
        )

        search_end = min(
            len(signal),
            search_end,
        )

        if search_end - search_start < self.minimum_wave_samples:
            return None, None, None

        result = self._find_wave(
            signal=signal,
            search_start=search_start,
            search_end=search_end,
            amplitude_threshold_ratio=(self.t_amplitude_threshold_ratio),
            noise_multiplier=(self.t_noise_multiplier),
            persistence=max(
                1, int(round(self.t_boundary_persistence_s * signal.sample_rate))
            ),
        )

        return (
            result.onset,
            result.peak,
            result.offset,
        )

    def delineate(
        self, signal: ECGSignal,
        r_samples: list[int]
    ) -> list[ECGAnnotation]:
        if r_samples.ndim != 1:
            raise ValueError("r_samples must be one-dimensional.")

        if len(r_samples) == 0:
            return ECGAnnotation.from_points({})

        if not np.all(np.isfinite(r_samples)):
            raise ValueError("r_samples contains non-finite values.")

        if np.any(np.diff(r_samples) <= 0):
            raise ValueError("r_samples must be strictly increasing.")

        if np.any(r_samples < 0) or np.any(r_samples >= len(signal.sample)):
            raise ValueError("r_samples contains out-of-range samples.")

        qrs_filtered = butterworth_filter(
            signal=signal,
            low_cutoff=self.qrs_low_cutoff,
            high_cutoff=self.qrs_high_cutoff,
            order=self.qrs_order,
        )

        p_filtered = butterworth_filter(
            signal=signal,
            low_cutoff=self.p_low_cutoff,
            high_cutoff=self.p_high_cutoff,
            order=self.p_order,
        )

        t_filtered = butterworth_filter(
            signal=signal,
            low_cutoff=self.t_low_cutoff,
            high_cutoff=self.t_high_cutoff,
            order=self.t_order,
        )

        points: dict[Fiducials, list[int]] = {
            fiducial: [] for fiducial in self.supported_points
        }

        for i, r_value in enumerate(r_samples):
            r = int(r_value)

            prev_r = int(r_samples[i - 1]) if i > 0 else None

            next_r = int(r_samples[i + 1]) if i + 1 < len(r_samples) else None

            qrs_onset, q_peak, s_peak, qrs_offset = self._delineate_qrs_samples(
                qrs_filtered, current_r=r
            )

            p_onset, p_peak, p_offset = self._delineate_p_samples(
                p_filtered,
                boundary_end=(qrs_onset if qrs_onset is not None else r),
                prev_r=prev_r,
            )

            t_onset, t_peak, t_offset = self._delineate_t_samples(
                t_filtered,
                boundary_start=(qrs_offset if qrs_offset is not None else r),
                current_r=r,
                next_r=next_r,
            )

            for sample, fiducial in (
                (p_onset, Fiducials.P_ONSET),
                (p_peak, Fiducials.P_PEAK),
                (p_offset, Fiducials.P_OFFSET),
                (qrs_onset, Fiducials.QRS_ONSET),
                (q_peak, Fiducials.Q_PEAK),
                (s_peak, Fiducials.S_PEAK),
                (qrs_offset, Fiducials.QRS_OFFSET),
                (t_onset, Fiducials.T_ONSET),
                (t_peak, Fiducials.T_PEAK),
                (t_offset, Fiducials.T_OFFSET),
            ):
                if sample is not None:
                    points[fiducial].append(int(sample))

        return ECGAnnotation.from_points(
            {
                fiducial: np.asarray(
                    samples,
                    dtype=np.int64,
                )
                for fiducial, samples in points.items()
            }
        )
