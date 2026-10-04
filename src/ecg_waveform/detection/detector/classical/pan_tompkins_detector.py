from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials
from ecg_waveform.signal.preprocessing import butterworth_filter

from ..base_detector import BaseDetector
from ..detector_registry import register_detector


@dataclass
class _Preprocessing:
    bandpassed: ECGSignal
    derivative: ECGSignal
    squared: ECGSignal
    smoothed: ECGSignal
    integrated: ECGSignal


@register_detector("pan_tompkins_plus_plus")
@dataclass(repr=False, slots=True)
class PanTompkinsPlusPlusDetector(BaseDetector):
    low_cutoff: float = 5.0
    high_cutoff: float = 18.0
    order: int = 4

    smoothing_window_s: float = 0.06
    integration_window_s: float = 0.15
    refractory_period_s: float = 0.231

    close_beat_period_s: float = 0.36
    slope_window_s: float = 0.07
    slope_ratio_threshold: float = 0.6

    searchback_gap_period_s: float = 1.0
    searchback_gap_ratio: float = 1.66
    weak_searchback_gap_period_s: float = 1.4
    weak_searchback_threshold_ratio: float = 0.2

    min_beats_for_rr_check: int = 8
    rr_history_size: int = 8
    threshold2_ratio: float = 0.4
    init_window_s: float = 2.0

    @property
    def supported_points(self) -> frozenset[Fiducials]:
        return frozenset({Fiducials.R_PEAK})

    def bandpass(self, signal: ECGSignal) -> ECGSignal:
        return butterworth_filter(
            signal=signal,
            low_cutoff=self.low_cutoff,
            high_cutoff=self.high_cutoff,
            order=self.order,
        )

    def derivative(self, signal: ECGSignal) -> ECGSignal:
        kernel = np.array([-1, -2, 0, 2, 1]) * (signal.sample_rate / 8)
        return signal.with_sample(np.convolve(signal.sample, kernel, mode="same"))

    def square(self, signal: ECGSignal) -> ECGSignal:
        return signal.with_sample(np.square(signal.sample))

    def smooth(self, signal: ECGSignal) -> ECGSignal:
        n_samples = max(3, int(self.smoothing_window_s * signal.sample_rate))
        window = np.ones(n_samples, dtype=float) / n_samples

        return signal.with_sample(np.convolve(signal.sample, window, mode="same"))

    def moving_window_integration(self, signal: ECGSignal) -> ECGSignal:
        n_samples = max(1, int(self.integration_window_s * signal.sample_rate))
        window = np.ones(n_samples) / n_samples

        return signal.with_sample(np.convolve(signal.sample, window, mode="same"))

    def preprocess(self, signal: ECGSignal) -> _Preprocessing:
        bandpassed = self.bandpass(signal)
        derivative = self.derivative(bandpassed)
        squared = self.square(derivative)
        smoothed = self.smooth(squared)
        integrated = self.moving_window_integration(smoothed)

        return _Preprocessing(
            bandpassed=bandpassed,
            derivative=derivative,
            squared=squared,
            smoothed=smoothed,
            integrated=integrated,
        )

    def _refine_r_peak(
        self, signal: np.ndarray, sample_rate: int, candidate: int
    ) -> int:
        search_left = max(0, candidate - int(0.08 * sample_rate))
        search_right = min(len(signal), candidate + int(0.12 * sample_rate))
        segment = signal[search_left:search_right]

        if len(segment) < 3:
            return candidate

        positive_peaks, _ = find_peaks(segment)
        negative_peaks, _ = find_peaks(-segment)
        candidate_peaks = np.concatenate([positive_peaks, negative_peaks])

        if len(candidate_peaks) == 0:
            return candidate

        candidate_local = candidate - search_left
        best_peak = None
        best_score = -np.inf

        for peak_idx in candidate_peaks:
            amplitude = abs(segment[peak_idx])
            distance = abs(peak_idx - candidate_local)
            score = amplitude - 0.05 * distance

            if score > best_score:
                best_score = score
                best_peak = peak_idx

        return search_left + best_peak

    def _mean_slope(self, signal: np.ndarray, sample_rate: int, index: int) -> float:
        window = max(1, int(self.slope_window_s * sample_rate))
        left = max(0, index - window)
        segment = signal[left : index + 1]
        if len(segment) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(segment))))

    def _mean_searchback(
        self,
        integrated: np.ndarray,
        qrs_peaks: list[int],
        candidate_peaks: np.ndarray,
        current_index: int,
    ) -> float:
        recent_qrs_values = [float(integrated[p]) for p in qrs_peaks[-3:]]

        following_values: list[float] = []
        last_qrs = qrs_peaks[-1] if qrs_peaks else None
        for peak in candidate_peaks[:current_index]:
            if last_qrs is not None and peak <= last_qrs:
                continue
            following_values.append(float(integrated[peak]))
            if len(following_values) >= 3:
                break

        values = recent_qrs_values + following_values
        if not values:
            return 0.0
        return float(np.mean(values))

    def _detect_r_samples(
        self,
        bandpassed: np.ndarray,
        integrated: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        refractory_period = int(self.refractory_period_s * sample_rate)
        candidate_peaks, _ = find_peaks(integrated, distance=refractory_period)

        if len(candidate_peaks) == 0:
            return np.array([], dtype=np.int64)

        init_samples = int(self.init_window_s * sample_rate)
        init_segment = integrated[:init_samples] if init_samples > 0 else integrated
        if len(init_segment) == 0:
            init_segment = integrated

        MAXF = float(np.max(init_segment))
        MEANF = float(np.mean(init_segment))

        THR1 = MAXF / 3 if MAXF > 0 else 0.0
        THR2 = 0.5 * MEANF
        SPKI = THR1
        NPKI = THR2

        qrs_peaks: list[int] = []
        rr_intervals: list[int] = []
        last_qrs: int | None = None

        close_beat_samples = int(self.close_beat_period_s * sample_rate)
        searchback_gap_samples = int(self.searchback_gap_period_s * sample_rate)
        weak_searchback_gap_samples = int(
            self.weak_searchback_gap_period_s * sample_rate
        )

        for idx, peak in enumerate(candidate_peaks):
            value = float(integrated[peak])
            accept = False
            accepted_location = peak
            rule = 1

            if value > THR1:
                accept = True
                rule = 1

                if (
                    len(qrs_peaks) > self.min_beats_for_rr_check
                    and last_qrs is not None
                ):
                    current_rr = peak - last_qrs
                    recent_rr = rr_intervals[-self.rr_history_size :]
                    mean_rr = np.mean(recent_rr) if recent_rr else current_rr

                    if current_rr < close_beat_samples or current_rr < 0.5 * mean_rr:
                        curr_slope = self._mean_slope(bandpassed, sample_rate, peak)
                        prev_slope = self._mean_slope(bandpassed, sample_rate, last_qrs)
                        is_t_wave = curr_slope < self.slope_ratio_threshold * prev_slope
                        accept = not is_t_wave

            elif last_qrs is not None:
                current_gap = peak - last_qrs
                recent_rr = rr_intervals[-self.rr_history_size :]
                mean_rr = float(np.mean(recent_rr)) if recent_rr else None

                search_start = last_qrs + close_beat_samples
                search_end = peak

                if (
                    search_end > search_start
                    and mean_rr is not None
                    and (
                        current_gap > searchback_gap_samples
                        or current_gap > self.searchback_gap_ratio * mean_rr
                    )
                ):
                    window = integrated[search_start:search_end]
                    MEANSB = self._mean_searchback(
                        integrated, qrs_peaks, candidate_peaks, idx
                    )
                    THR3 = 0.5 * THR2 + 0.5 * MEANSB
                    local_max = int(np.argmax(window))
                    if window[local_max] > THR3:
                        accept = True
                        rule = 2
                        accepted_location = search_start + local_max

                elif (
                    search_end > search_start
                    and current_gap > weak_searchback_gap_samples
                ):
                    window = integrated[search_start:search_end]
                    local_max = int(np.argmax(window))
                    if window[local_max] > self.weak_searchback_threshold_ratio * THR2:
                        accept = True
                        rule = 2
                        accepted_location = search_start + local_max

            if accept and qrs_peaks:
                if accepted_location - qrs_peaks[-1] < refractory_period:
                    accept = False

            if accept:
                refined = self._refine_r_peak(
                    bandpassed, sample_rate, accepted_location
                )

                if rule == 1:
                    SPKI = 0.125 * value + 0.875 * SPKI
                else:
                    SPKI = 0.75 * value + 0.25 * SPKI

                if last_qrs is not None:
                    rr = refined - last_qrs
                    if rr > 0:
                        rr_intervals.append(rr)
                        if len(rr_intervals) > self.rr_history_size:
                            rr_intervals.pop(0)

                qrs_peaks.append(refined)
                last_qrs = refined
            else:
                if rule == 1:
                    NPKI = 0.125 * value + 0.875 * NPKI
                else:
                    NPKI = 0.75 * value + 0.25 * NPKI

            THR1 = NPKI + 0.25 * (SPKI - NPKI)
            THR2 = self.threshold2_ratio * THR1

        return np.array(sorted(qrs_peaks), dtype=np.int64)

    def detect(self, signal: ECGSignal) -> ECGAnnotation:
        preprocessed = self.preprocess(signal)
        bandpassed = preprocessed.bandpassed.sample
        integrated = preprocessed.integrated.sample

        r_samples = self._detect_r_samples(bandpassed, integrated, signal.sample_rate)
        return ECGAnnotation(
            symbol=np.array([Fiducials.R_PEAK.value] * len(r_samples), dtype=str),
            sample=r_samples,
        )
