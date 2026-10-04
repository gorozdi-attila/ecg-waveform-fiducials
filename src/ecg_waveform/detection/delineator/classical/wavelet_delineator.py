from dataclasses import dataclass

import numpy as np

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials

from ...wavelet_common import (
    DEFAULT_PEAK_REFINE_S,
    DEFAULT_QRS_THRESHOLD_FACTORS,
    DEFAULT_RMS_WINDOW_S,
    MIN_WORKING_SAMPLES,
    WORKING_RATE,
    Analysis,
    localize_peak,
    samples,
    significant_crossings,
    threshold_crossing,
    to_original_sample,
    to_working_rate,
    zero_crossings,
)
from ..base_delineator import BaseDelineator
from ..delineator_registry import register_delineator


@dataclass(slots=True)
class _Wave:
    onset: int
    peak: float
    offset: int
    peak_is_maximum: bool


@dataclass(slots=True)
class _Beat:
    r: float
    r_is_maximum: bool
    pair: tuple[int, int]
    qrs_onset: int | None = None
    qrs_offset: int | None = None
    q: float | None = None
    s: float | None = None
    t: _Wave | None = None
    p: _Wave | None = None


@register_delineator("martinez_wavelet")
@dataclass(repr=False, slots=True)
class MartinezWaveletDelineator(BaseDelineator):
    rms_window_s: float = DEFAULT_RMS_WINDOW_S
    threshold_factors: tuple[float, ...] = DEFAULT_QRS_THRESHOLD_FACTORS

    qrs_pair_reach_s: float = 0.08
    qrs_wave_fraction: float = 0.1
    qrs_chain_gap_s: float = 0.04
    qrs_boundary_search_s: float = 0.1
    qrs_onset_fraction: float = 0.05
    qrs_offset_fraction: float = 0.125

    wave_scales: tuple[int, ...] = (4, 5)
    wave_pair_max_s: float = 0.25
    wave_chain_gap_s: float = 0.15
    wave_boundary_search_s: float = 0.2
    minimum_window_s: float = 0.05

    p_amplitude_ratio: float = 0.03
    p_onset_fraction: float = 0.5
    p_offset_fraction: float = 0.5
    p_search_before_s: float = 0.225
    p_search_gap_s: float = 0.04

    t_amplitude_ratio: float = 0.06
    t_onset_fraction: float = 0.25
    t_offset_fraction: float = 0.4
    t_search_gap_s: float = 0.1
    t_search_after_s: float = 0.5
    t_max_rr_fraction: float = 0.65
    t_default_gap_s: float = 0.6

    peak_refine_s: float = DEFAULT_PEAK_REFINE_S

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

    def _start_beat(
        self,
        analysis: Analysis,
        thresholds: dict[int, np.ndarray],
        r_sample: int,
        rate_ratio: float,
    ) -> _Beat | None:
        w2 = analysis.wavelet_coefficients[2]
        maxima = analysis.modulus_maxima[2]
        strong = maxima[np.abs(w2[maxima]) >= thresholds[2][maxima]]

        r = r_sample / rate_ratio
        center = int(round(r))
        reach = samples(self.qrs_pair_reach_s)

        left = strong[(strong >= center - reach) & (strong <= center)]
        if len(left) == 0:
            return None
        first = int(left[-1])

        right = strong[(strong > center) & (strong <= center + reach)]
        right = [int(p) for p in right if np.sign(w2[p]) != np.sign(w2[first])]
        if not right:
            return None

        return _Beat(r=r, r_is_maximum=bool(w2[first] > 0), pair=(first, right[0]))

    def _outermost_maximum(
        self, analysis: Analysis, start: int, direction: int, reference: float
    ) -> int:
        w2 = analysis.wavelet_coefficients[2]
        maxima = analysis.modulus_maxima[2]
        gap = samples(self.qrs_chain_gap_s)
        limit = samples(self.qrs_boundary_search_s)

        current = start
        while True:
            if direction < 0:
                pool = maxima[(maxima < current) & (maxima >= current - gap)]
            else:
                pool = maxima[(maxima > current) & (maxima <= current + gap)]
            pool = [
                int(p) for p in pool if abs(w2[p]) >= self.qrs_wave_fraction * reference
            ]
            if not pool:
                return current

            nearest = pool[0] if direction > 0 else pool[-1]
            if abs(nearest - start) > limit:
                return current
            current = nearest

    def _delineate_qrs(
        self, analysis: Analysis, thresholds: dict[int, np.ndarray], beat: _Beat
    ) -> bool:
        w2 = analysis.wavelet_coefficients[2]
        first_pair, last_pair = beat.pair

        reference = max(abs(w2[first_pair]), abs(w2[last_pair]))
        first = self._outermost_maximum(analysis, first_pair, -1, reference)
        last = self._outermost_maximum(analysis, last_pair, +1, reference)

        limit = samples(self.qrs_boundary_search_s)
        onset = threshold_crossing(w2, first, -1, self.qrs_onset_fraction, limit)
        offset = threshold_crossing(w2, last, +1, self.qrs_offset_fraction, limit)
        if not onset < beat.r < offset:
            return False
        beat.qrs_onset, beat.qrs_offset = onset, offset

        crossings = significant_crossings(
            analysis, thresholds, 1, onset - 1, offset + 1, falling=not beat.r_is_maximum
        )
        before = [c for c in crossings if c < beat.r - 1]
        after = [c for c in crossings if c > beat.r + 1]
        beat.q = before[-1] if before else None
        beat.s = after[0] if after else None
        return True

    def _wave_at_scale(
        self,
        analysis: Analysis,
        scale: int,
        start: int,
        stop: int,
        amplitude_threshold: float,
        onset_fraction: float,
        offset_fraction: float,
    ) -> _Wave | None:
        coefficients = analysis.wavelet_coefficients[scale]
        maxima = analysis.modulus_maxima[scale]
        significant = maxima[(maxima >= start) & (maxima < stop)]
        significant = significant[np.abs(coefficients[significant]) >= amplitude_threshold]
        if len(significant) < 2:
            return None

        signs = np.sign(coefficients[significant])
        moduli = np.abs(coefficients[significant])

        pair_max = samples(self.wave_pair_max_s)
        best, best_score = None, 0.0
        for i in range(len(significant) - 1):
            close = significant[i + 1] - significant[i] <= pair_max
            if signs[i] != signs[i + 1] and close and moduli[i] + moduli[i + 1] > best_score:
                best, best_score = i, moduli[i] + moduli[i + 1]
        if best is None:
            return None

        chain_gap = samples(self.wave_chain_gap_s)
        first, last = best, best + 1
        before = (
            first - 1
            if first > 0
            and signs[first - 1] != signs[first]
            and significant[first] - significant[first - 1] <= chain_gap
            else None
        )
        after = (
            last + 1
            if last + 1 < len(significant)
            and signs[last + 1] != signs[last]
            and significant[last + 1] - significant[last] <= chain_gap
            else None
        )
        if before is not None and (after is None or moduli[before] >= moduli[after]):
            first = before
        elif after is not None:
            last = after

        peak_is_maximum = bool(signs[best] > 0)
        crossings = zero_crossings(
            coefficients, significant[best], significant[best + 1], falling=peak_is_maximum
        )
        if not crossings:
            return None

        limit = samples(self.wave_boundary_search_s)
        return _Wave(
            onset=threshold_crossing(coefficients, int(significant[first]), -1, onset_fraction, limit),
            peak=crossings[0][1],
            offset=threshold_crossing(coefficients, int(significant[last]), +1, offset_fraction, limit),
            peak_is_maximum=peak_is_maximum,
        )

    def _find_wave(
        self,
        analysis: Analysis,
        beat: _Beat,
        start: int,
        stop: int,
        amplitude_ratio: float,
        onset_fraction: float,
        offset_fraction: float,
    ) -> _Wave | None:
        if stop - start < samples(self.minimum_window_s):
            return None

        for scale in self.wave_scales:
            coefficients = analysis.wavelet_coefficients[scale]
            qrs_modulus = float(
                np.max(np.abs(coefficients[beat.qrs_onset : beat.qrs_offset + 1]))
            )
            wave = self._wave_at_scale(
                analysis, scale, start, stop, amplitude_ratio * qrs_modulus,
                onset_fraction, offset_fraction,
            )
            if wave is not None:
                return wave
        return None

    def _delineate_t(self, analysis: Analysis, beats: list[_Beat], index: int) -> None:
        beat = beats[index]
        following = beats[index + 1] if index + 1 < len(beats) else None

        gap = (
            following.qrs_onset - beat.qrs_offset
            if following is not None
            else samples(self.t_default_gap_s)
        )
        start = beat.qrs_offset + samples(self.t_search_gap_s)
        stop = beat.qrs_offset + min(
            samples(self.t_search_after_s), int(self.t_max_rr_fraction * gap)
        )

        wave = self._find_wave(
            analysis, beat, start, min(stop, analysis.n_samples),
            self.t_amplitude_ratio, self.t_onset_fraction, self.t_offset_fraction,
        )
        if wave is not None:
            wave.onset = max(wave.onset, beat.qrs_offset + 1)
            beat.t = wave

    def _delineate_p(self, analysis: Analysis, beats: list[_Beat], index: int) -> None:
        beat = beats[index]

        floor = 0
        if index > 0:
            previous = beats[index - 1]
            floor = (
                previous.t.offset
                if previous.t is not None
                else previous.qrs_offset + samples(self.t_search_gap_s)
            )

        start = max(beat.qrs_onset - samples(self.p_search_before_s), floor, 0)
        stop = beat.qrs_onset - samples(self.p_search_gap_s)

        wave = self._find_wave(
            analysis, beat, start, stop,
            self.p_amplitude_ratio, self.p_onset_fraction, self.p_offset_fraction,
        )
        if wave is not None:
            wave.offset = min(wave.offset, beat.qrs_onset - 1)
            beat.p = wave

    def _to_annotation(
        self, original: np.ndarray, beats: list[_Beat], rate_ratio: float
    ) -> ECGAnnotation:
        n = len(original)

        def boundary(position: float) -> int:
            return to_original_sample(position, rate_ratio, n)

        def peak(position: float, is_maximum: bool) -> int:
            return localize_peak(original, position, is_maximum, rate_ratio, self.peak_refine_s)

        points: dict[Fiducials, list[int]] = {f: [] for f in self.supported_points}
        for beat in beats:
            points[Fiducials.QRS_ONSET].append(boundary(beat.qrs_onset))
            points[Fiducials.QRS_OFFSET].append(boundary(beat.qrs_offset))
            if beat.q is not None:
                points[Fiducials.Q_PEAK].append(peak(beat.q, not beat.r_is_maximum))
            if beat.s is not None:
                points[Fiducials.S_PEAK].append(peak(beat.s, not beat.r_is_maximum))

            for wave, onset, top, offset in (
                (beat.t, Fiducials.T_ONSET, Fiducials.T_PEAK, Fiducials.T_OFFSET),
                (beat.p, Fiducials.P_ONSET, Fiducials.P_PEAK, Fiducials.P_OFFSET),
            ):
                if wave is None:
                    continue
                points[top].append(peak(wave.peak, wave.peak_is_maximum))
                if wave.onset < wave.peak:
                    points[onset].append(boundary(wave.onset))
                if wave.offset > wave.peak:
                    points[offset].append(boundary(wave.offset))

        return ECGAnnotation.from_points(
            {f: np.asarray(v, dtype=np.int64) for f, v in points.items()}
        )

    def delineate(
        self, signal: ECGSignal, r_samples: np.ndarray | ECGAnnotation
    ) -> ECGAnnotation:
        if isinstance(r_samples, ECGAnnotation):
            r_samples = r_samples.filter(Fiducials.R_PEAK.value).sample
        r_samples = np.asarray(r_samples)

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

        original = np.asarray(signal.sample, dtype=np.float64)
        rate_ratio = signal.sample_rate / WORKING_RATE

        working = to_working_rate(original, signal.sample_rate)
        if len(working) < MIN_WORKING_SAMPLES:
            return ECGAnnotation.from_points({})

        analysis = Analysis.from_ecg(working, self.rms_window_s)
        thresholds = analysis.thresholds(self.threshold_factors)

        beats: list[_Beat] = []
        for r in r_samples:
            beat = self._start_beat(analysis, thresholds, int(r), rate_ratio)
            if beat is not None and self._delineate_qrs(analysis, thresholds, beat):
                beats.append(beat)

        for index in range(len(beats)):
            self._delineate_t(analysis, beats, index)
        for index in range(len(beats)):
            self._delineate_p(analysis, beats, index)

        return self._to_annotation(original, beats, rate_ratio)