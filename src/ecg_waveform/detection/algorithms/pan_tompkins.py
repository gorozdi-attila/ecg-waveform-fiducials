from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from ..base import BaseDetector
from ..delineator.base import BaseDelineator
from ..delineator.windowed import WindowedFiducialDelineator
from ..registry import register_detector

from ecg_waveform.core import ECGSignal, ECGAnnotation, Fiducials
from ecg_waveform.signal.preprocessing import butterworth_filter


@dataclass
class PreprocessedSignal:
    bandpassed: ECGSignal
    derivative: ECGSignal
    squared: ECGSignal
    integrated: ECGSignal


@dataclass(repr=False)
class PanTompkinsDetector(BaseDetector):
    high_cutoff: float = 15.0
    low_cutoff: float = 5.0
    order: int = 4
    window_size_s: float = 0.15
    refractory_period_s: float = 0.2

    delineator: BaseDelineator | None = field(default_factory=WindowedFiducialDelineator)

    @property
    def supported_points(self) -> frozenset[Fiducials]:
        if self.delineator is None:
            return frozenset({Fiducials.R_WAVE})
        return frozenset(Fiducials)

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

    def moving_average(self, signal: ECGSignal) -> ECGSignal:
        moving_window = int(self.window_size_s * signal.sample_rate)
        window = np.ones(moving_window) / moving_window
        return signal.with_sample(np.convolve(signal.sample, window, mode="same"))

    def preprocess(self, signal: ECGSignal) -> PreprocessedSignal:
        bandpassed = self.bandpass(signal)
        derivative = self.derivative(bandpassed)
        squared = self.square(derivative)
        integrated = self.moving_average(squared)

        return PreprocessedSignal(
            bandpassed=bandpassed,
            derivative=derivative,
            squared=squared,
            integrated=integrated,
        )

    def _refine_r_peaks(
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

    def _detect_r_samples(
        self, bandpassed: np.ndarray, integrated: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        refractory_period = int(self.refractory_period_s * sample_rate)

        bandpassed_peaks, _ = find_peaks(np.abs(bandpassed), distance=refractory_period)
        integrated_peaks, _ = find_peaks(integrated, distance=refractory_period)

        candidate_peaks = np.unique(
            np.concatenate([bandpassed_peaks, integrated_peaks])
        )
        candidate_peaks.sort()

        if len(candidate_peaks) == 0:
            return np.array([], dtype=np.int64)

        initial_values = integrated[candidate_peaks[: min(12, len(candidate_peaks))]]

        SPKI = np.percentile(initial_values, 95)
        NPKI = np.percentile(initial_values, 20)

        THR1 = NPKI + 0.25 * (SPKI - NPKI)
        THR2 = 0.5 * THR1

        qrs_peaks: list[int] = []
        rr_intervals: list[int] = []

        last_qrs = None
        searchback_candidates: list[tuple[int, float]] = []

        def rr_expected() -> float | None:
            if len(rr_intervals) < 3:
                return None
            return np.median(rr_intervals)

        for peak in candidate_peaks:
            value = integrated[peak]
            is_qrs = False

            if THR2 < value < THR1:
                searchback_candidates.append((peak, value))

            if value >= THR1:
                is_qrs = True

            rr_avg = rr_expected()

            if not is_qrs and last_qrs is not None and rr_avg is not None:
                if peak - last_qrs > 1.6 * rr_avg and searchback_candidates:
                    peak, value = max(searchback_candidates, key=lambda x: x[1])
                    is_qrs = True

            if is_qrs:
                refined_peak = self._refine_r_peaks(bandpassed, sample_rate, peak)

                if qrs_peaks and refined_peak - qrs_peaks[-1] < refractory_period:
                    continue

                qrs_peaks.append(refined_peak)
                SPKI = 0.125 * value + 0.875 * SPKI
                searchback_candidates = []

                if last_qrs is not None:
                    rr = refined_peak - last_qrs
                    if rr > 0:
                        rr_intervals.append(rr)
                        if len(rr_intervals) > 8:
                            rr_intervals.pop(0)

                last_qrs = refined_peak
            else:
                NPKI = 0.125 * value + 0.875 * NPKI

            THR1 = NPKI + 0.22 * (SPKI - NPKI)
            THR2 = 0.5 * THR1

        return np.array(qrs_peaks, dtype=np.int64)

    def detect_r_point(self, signal: ECGSignal) -> ECGAnnotation:
        return self.detect_points(signal, points=[Fiducials.R_WAVE])

    def detect_q_point(self, signal: ECGSignal) -> ECGAnnotation:
        return self.detect_points(signal, points=[Fiducials.Q_WAVE])

    def detect_s_point(self, signal: ECGSignal) -> ECGAnnotation:
        return self.detect_points(signal, points=[Fiducials.S_WAVE])

    def detect_p_point(self, signal: ECGSignal) -> ECGAnnotation:
        return self.detect_points(signal, points=[Fiducials.P_WAVE])

    def detect_t_point(self, signal: ECGSignal) -> ECGAnnotation:
        return self.detect_points(signal, points=[Fiducials.T_WAVE])

    def detect(self, signal: ECGSignal) -> ECGAnnotation:
        preprocessed = self.preprocess(signal)
        bandpassed = preprocessed.bandpassed.sample
        integrated = preprocessed.integrated.sample

        r_samples = self._detect_r_samples(bandpassed, integrated, signal.sample_rate)
        r_annotation = ECGAnnotation(
            symbol=np.array([Fiducials.R_WAVE.value] * len(r_samples), dtype=str),
            sample=r_samples,
        )

        if self.delineator is None:
            return r_annotation

        fiducials = self.delineator.delineate(signal, r_samples, qrs_band=bandpassed)
        return r_annotation.merge(fiducials)


register_detector("pan_tompkins", PanTompkinsDetector)
