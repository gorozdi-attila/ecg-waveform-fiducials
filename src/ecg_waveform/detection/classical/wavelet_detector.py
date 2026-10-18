from dataclasses import dataclass
from itertools import pairwise
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials

from ..base_detector import BasePQRSTDetector
from ..registry import register_detector

_WORKING_RATE = 250
_SCALES = (1, 2, 3, 4, 5)
_LOWPASS = np.array([1.0, 3.0, 3.0, 1.0]) / 8.0
_HIGHPASS = np.array([2.0, -2.0])

_QRS_RMS_FACTORS = {1: 0.5, 2: 0.5, 3: 0.6, 4: 0.8}
_WAVE_SCALES = (4, 5)

_MIN_WORKING_SAMPLES = 2 ** (len(_SCALES) + 1) * 4


def _samples(seconds: float) -> int:
    return int(seconds * _WORKING_RATE)


def _upsample_filter(taps: np.ndarray, factor: int) -> np.ndarray:
    upsampled = np.zeros((len(taps) - 1) * factor + 1)
    upsampled[::factor] = taps
    return upsampled


def _wavelet_transform(ecg: np.ndarray) -> dict[int, np.ndarray]:
    pad = 2 ** (len(_SCALES) + 1)
    approximation = np.pad(ecg, pad, mode="edge")
    coefficients: dict[int, np.ndarray] = {}
    for k in _SCALES:
        dilation = 2 ** (k - 1)
        detail = np.convolve(approximation, _upsample_filter(_HIGHPASS, dilation))
        approximation = np.convolve(approximation, _upsample_filter(_LOWPASS, dilation))
        start = pad + 2**k - 2
        coefficients[k] = detail[start : start + len(ecg)]
    return coefficients


def _find_modulus_maxima(wavelet_coefficients: np.ndarray) -> np.ndarray:
    modulus = np.abs(wavelet_coefficients)
    is_maximum = (
        (modulus[1:-1] >= modulus[:-2])
        & (modulus[1:-1] > modulus[2:])
        & (modulus[1:-1] > 0)
    )
    return np.flatnonzero(is_maximum) + 1


def _moving_rms(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values**2)))
    index = np.arange(len(values))
    low = np.clip(index - window // 2, 0, len(values))
    high = np.clip(index + window // 2 + 1, 0, len(values))
    return np.sqrt((cumulative[high] - cumulative[low]) / (high - low))


@dataclass
class _Analysis:
    wavelet_coefficients: dict[int, np.ndarray]
    modulus_maxima: dict[int, np.ndarray]
    local_rms: dict[int, np.ndarray]

    @classmethod
    def from_ecg(cls, ecg: np.ndarray, rms_window_s: float) -> "_Analysis":
        coefficients = _wavelet_transform(ecg)
        return cls(
            wavelet_coefficients=coefficients,
            modulus_maxima={
                k: _find_modulus_maxima(w) for k, w in coefficients.items()
            },
            local_rms={
                k: _moving_rms(w, _samples(rms_window_s))
                for k, w in coefficients.items()
            },
        )

    @property
    def n_samples(self) -> int:
        return len(self.wavelet_coefficients[1])

    def qrs_thresholds(self, scale_factor: float = 1.0) -> dict[int, np.ndarray]:
        return {
            k: scale_factor * factor * self.local_rms[k]
            for k, factor in _QRS_RMS_FACTORS.items()
        }

    def qrs_threshold_at(self, scale: int, index: int) -> float:
        return _QRS_RMS_FACTORS[scale] * float(self.local_rms[scale][index])


def _zero_crossings(
    coefficients: np.ndarray, low: int, high: int, falling: bool
) -> list[tuple[int, float]]:
    start = max(low, 0)
    stop = min(high, len(coefficients) - 1)
    if stop <= start:
        return []
    values = coefficients[start : stop + 1].tolist()
    crossings = []
    for offset in range(stop - start):
        a, b = values[offset], values[offset + 1]
        hit = (a > 0 and b <= 0) if falling else (a < 0 and b >= 0)
        if hit:
            n = start + offset
            crossings.append((n, n - 0.5 + a / (a - b)))
    return crossings


def _significant_crossings(
    analysis: _Analysis, scale: int, low: int, high: int, falling: bool
) -> list[float]:
    coefficients = analysis.wavelet_coefficients[scale]
    maxima = analysis.modulus_maxima[scale]
    locations = []
    for n, location in _zero_crossings(coefficients, low, high, falling):
        right = int(np.searchsorted(maxima, n, side="right"))
        if right == 0 or right >= len(maxima):
            continue
        left_max, right_max = maxima[right - 1], maxima[right]
        if (
            abs(coefficients[left_max])
            >= analysis.qrs_threshold_at(scale, int(left_max))
            and abs(coefficients[right_max])
            >= analysis.qrs_threshold_at(scale, int(right_max))
        ):
            locations.append(location)
    return locations


def _threshold_crossing(
    coefficients: np.ndarray, start: int, step: int, fraction: float, limit: int
) -> int:
    size = len(coefficients)
    lo = max(0, start - limit - 1) if step < 0 else start
    hi = min(size, start + limit + 2) if step > 0 else start + 1
    window = np.abs(coefficients[lo:hi]).tolist()
    threshold = fraction * window[start - lo]
    n = start
    for _ in range(limit):
        following = n + step
        if following < 0 or following >= size:
            break
        value = window[following - lo]
        if value < threshold:
            return following
        if value > window[n - lo]:
            return n
        n = following
    return n


def _link_maxima(
    analysis: _Analysis,
    start_scale: int,
    start_position: int,
    thresholds: dict[int, np.ndarray],
) -> dict[int, int]:
    coefficients = analysis.wavelet_coefficients
    sign = np.sign(coefficients[start_scale][start_position])
    line = {start_scale: start_position}
    current = start_position
    for scale in range(start_scale - 1, 0, -1):
        maxima = analysis.modulus_maxima[scale]
        radius = 2**scale + 2
        lo = int(np.searchsorted(maxima, current - radius, side="left"))
        hi = int(np.searchsorted(maxima, current + radius, side="right"))
        near = [
            int(p)
            for p in maxima[lo:hi]
            if np.sign(coefficients[scale][p]) == sign
            and abs(coefficients[scale][p]) >= thresholds[scale][p]
        ]
        if near:
            current = min(near, key=lambda p: abs(p - current))
            line[scale] = current
    return line


@dataclass
class _Wave:
    onset: int
    peak: float
    offset: int
    peak_is_maximum: bool


@dataclass
class _Beat:
    r: float
    r_is_maximum: bool
    strength: float
    pair: tuple[int, int]
    qrs_onset: int | None = None
    qrs_offset: int | None = None
    q: float | None = None
    s: float | None = None
    t: _Wave | None = None
    p: _Wave | None = None


@register_detector("wavelet")
@dataclass(repr=False, slots=True)
class WaveletDetector(BasePQRSTDetector):
    rms_window_s: float = 2.0

    qrs_pair_max_s: float = 0.15  # max separation of the +/- maxima pair at 2^4
    qrs_min_distance_s: float = 0.2  # redundant detection removal
    qrs_min_scales: int = 3  # a maxima line must persist over >= 3 of 4 scales
    missed_beat_rr_ratio: float = 1.5
    missed_beat_threshold_scale: float = 0.5
    qrs_wave_fraction: float = 0.1  # significance of extra maxima at 2^2 (Q/S waves)
    qrs_chain_gap_s: float = 0.04
    qrs_boundary_search_s: float = 0.1
    qrs_onset_fraction: float = 0.05
    qrs_offset_fraction: float = 0.125

    t_amplitude_fraction: float = 0.06
    t_onset_fraction: float = 0.25
    t_offset_fraction: float = 0.4
    t_window_start_s: float = 0.1
    t_window_max_s: float = 0.5
    t_window_rr_fraction: float = 0.65

    p_amplitude_fraction: float = 0.03
    p_onset_fraction: float = 0.5
    p_offset_fraction: float = 0.5
    p_window_start_s: float = 0.225  # before QRS onset
    p_window_end_s: float = 0.04  # before QRS onset

    wave_pair_max_s: float = 0.25
    wave_chain_gap_s: float = 0.15
    wave_boundary_search_s: float = 0.2

    peak_refine_s: float = 0.012

    @property
    def supported_points(self) -> frozenset[Fiducials]:
        return frozenset(
            {
                Fiducials.P_ONSET,
                Fiducials.P_PEAK,
                Fiducials.P_OFFSET,
                Fiducials.QRS_ONSET,
                Fiducials.Q_PEAK,
                Fiducials.R_PEAK,
                Fiducials.S_PEAK,
                Fiducials.QRS_OFFSET,
                Fiducials.T_ONSET,
                Fiducials.T_PEAK,
                Fiducials.T_OFFSET,
            }
        )

    def detect(self, signal: ECGSignal) -> ECGAnnotation:
        original = np.asarray(signal.sample, dtype=np.float64)
        sample_rate = signal.sample_rate
        working = self._to_working_rate(original, sample_rate)
        if len(working) < _MIN_WORKING_SAMPLES:
            return ECGAnnotation.from_points({})

        analysis = _Analysis.from_ecg(working, self.rms_window_s)
        beats = self._detect_qrs_complexes(analysis)
        for beat in beats:
            self._delineate_qrs(analysis, beat)
        beats = [b for b in beats if b.qrs_onset < b.r < b.qrs_offset]
        for index, beat in enumerate(beats):
            self._delineate_t_wave(analysis, beats, index)
        for index, beat in enumerate(beats):
            self._delineate_p_wave(analysis, beats, index)

        return self._build_annotation(original, beats, sample_rate / _WORKING_RATE)

    @staticmethod
    def _to_working_rate(ecg: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == _WORKING_RATE:
            return ecg
        divisor = gcd(_WORKING_RATE, sample_rate)
        return resample_poly(ecg, _WORKING_RATE // divisor, sample_rate // divisor)

    def _detect_qrs_complexes(self, analysis: _Analysis) -> list[_Beat]:
        beats = self._qrs_candidates(analysis, 1.0, 0, analysis.n_samples)
        beats = self._merge_redundant(beats)

        if len(beats) >= 2:
            median_rr = float(np.median(np.diff([b.r for b in beats])))
            margin = _samples(self.qrs_min_distance_s)
            recovered: list[_Beat] = []
            for left, right in pairwise(beats):
                if right.r - left.r > self.missed_beat_rr_ratio * median_rr:
                    recovered += self._qrs_candidates(
                        analysis,
                        self.missed_beat_threshold_scale,
                        int(left.r) + margin,
                        int(right.r) - margin,
                    )
            beats = self._merge_redundant(beats + recovered)
        return beats

    def _qrs_candidates(
        self, analysis: _Analysis, threshold_scale: float, low: int, high: int
    ) -> list[_Beat]:
        coefficients = analysis.wavelet_coefficients
        thresholds = analysis.qrs_thresholds(threshold_scale)
        maxima_4 = analysis.modulus_maxima[4]
        coarse = [
            int(p)
            for p in maxima_4
            if low <= p < high and abs(coefficients[4][p]) >= thresholds[4][p]
        ]
        pair_max = _samples(self.qrs_pair_max_s)
        beats: list[_Beat] = []
        i = 0
        while i < len(coarse) - 1:
            first, second = coarse[i], coarse[i + 1]
            opposite = np.sign(coefficients[4][first]) != np.sign(
                coefficients[4][second]
            )
            beat = None
            if opposite and second - first <= pair_max:
                beat = self._link_qrs_pair(analysis, first, second, thresholds)
            if beat is not None:
                beats.append(beat)
                i += 2
            else:
                i += 1
        return beats

    def _link_qrs_pair(
        self,
        analysis: _Analysis,
        first: int,
        second: int,
        thresholds: dict[int, np.ndarray],
    ) -> _Beat | None:
        line_a = _link_maxima(analysis, 4, first, thresholds)
        line_b = _link_maxima(analysis, 4, second, thresholds)
        if min(len(line_a), len(line_b)) < self.qrs_min_scales:
            return None
        if 2 not in line_a or 2 not in line_b or line_a[2] >= line_b[2]:
            return None
        w2 = analysis.wavelet_coefficients[2]
        r_is_maximum = analysis.wavelet_coefficients[4][first] > 0
        crossings = _zero_crossings(w2, line_a[2], line_b[2], falling=r_is_maximum)
        if not crossings:
            return None
        return _Beat(
            r=crossings[0][1],
            r_is_maximum=bool(r_is_maximum),
            strength=float(abs(w2[line_a[2]]) + abs(w2[line_b[2]])),
            pair=(line_a[2], line_b[2]),
        )

    def _merge_redundant(self, beats: list[_Beat]) -> list[_Beat]:
        minimum_distance = _samples(self.qrs_min_distance_s)
        kept: list[_Beat] = []
        for beat in sorted(beats, key=lambda b: b.r):
            if kept and beat.r - kept[-1].r < minimum_distance:
                if beat.strength > kept[-1].strength:
                    kept[-1] = beat
            else:
                kept.append(beat)
        return kept

    def _delineate_qrs(self, analysis: _Analysis, beat: _Beat) -> None:
        w2 = analysis.wavelet_coefficients[2]
        first_pair, last_pair = beat.pair

        r_center = int(beat.r)
        fine = _significant_crossings(
            analysis, 1, r_center - 3, r_center + 3, falling=beat.r_is_maximum
        )
        if fine:
            beat.r = min(fine, key=lambda c: abs(c - beat.r))

        reference = max(abs(w2[first_pair]), abs(w2[last_pair]))
        first = self._outermost_maximum(analysis, first_pair, -1, reference)
        last = self._outermost_maximum(analysis, last_pair, +1, reference)
        limit = _samples(self.qrs_boundary_search_s)
        beat.qrs_onset = _threshold_crossing(
            w2, first, -1, self.qrs_onset_fraction, limit
        )
        beat.qrs_offset = _threshold_crossing(
            w2, last, +1, self.qrs_offset_fraction, limit
        )

        crossings = _significant_crossings(
            analysis,
            1,
            beat.qrs_onset - 1,
            beat.qrs_offset + 1,
            falling=not beat.r_is_maximum,
        )
        before = [c for c in crossings if c < beat.r - 1]
        after = [c for c in crossings if c > beat.r + 1]
        beat.q = before[-1] if before else None
        beat.s = after[0] if after else None

    def _outermost_maximum(
        self, analysis: _Analysis, start: int, direction: int, reference: float
    ) -> int:
        w2 = analysis.wavelet_coefficients[2]
        maxima = analysis.modulus_maxima[2]
        gap = _samples(self.qrs_chain_gap_s)
        limit = _samples(self.qrs_boundary_search_s)
        current = start
        while True:
            if direction < 0:
                lo = int(np.searchsorted(maxima, current - gap, side="left"))
                hi = int(np.searchsorted(maxima, current, side="left"))
            else:
                lo = int(np.searchsorted(maxima, current, side="right"))
                hi = int(np.searchsorted(maxima, current + gap, side="right"))
            pool = maxima[lo:hi]
            pool = [
                int(p) for p in pool if abs(w2[p]) >= self.qrs_wave_fraction * reference
            ]
            if not pool or abs(pool[0 if direction > 0 else -1] - start) > limit:
                return current
            current = pool[0] if direction > 0 else pool[-1]

    def _delineate_t_wave(
        self, analysis: _Analysis, beats: list[_Beat], index: int
    ) -> None:
        beat = beats[index]
        following = beats[index + 1] if index + 1 < len(beats) else None
        gap = (
            following.qrs_onset - beat.qrs_offset
            if following is not None
            else _samples(0.6)
        )
        low = beat.qrs_offset + _samples(self.t_window_start_s)
        high = beat.qrs_offset + min(
            _samples(self.t_window_max_s), int(self.t_window_rr_fraction * gap)
        )
        high = min(high, analysis.n_samples)
        wave = self._delineate_wave(
            analysis,
            beat,
            low,
            high,
            self.t_amplitude_fraction,
            self.t_onset_fraction,
            self.t_offset_fraction,
        )
        if wave is not None:
            wave.onset = max(wave.onset, beat.qrs_offset + 1)
            beat.t = wave

    def _delineate_p_wave(
        self, analysis: _Analysis, beats: list[_Beat], index: int
    ) -> None:
        beat = beats[index]
        floor = 0
        if index > 0:
            previous = beats[index - 1]
            floor = (
                previous.t.offset
                if previous.t is not None
                else previous.qrs_offset + _samples(self.t_window_start_s)
            )
        low = max(beat.qrs_onset - _samples(self.p_window_start_s), floor, 0)
        high = beat.qrs_onset - _samples(self.p_window_end_s)
        wave = self._delineate_wave(
            analysis,
            beat,
            low,
            high,
            self.p_amplitude_fraction,
            self.p_onset_fraction,
            self.p_offset_fraction,
        )
        if wave is not None:
            wave.offset = min(wave.offset, beat.qrs_onset - 1)
            beat.p = wave

    def _delineate_wave(
        self,
        analysis: _Analysis,
        beat: _Beat,
        low: int,
        high: int,
        amplitude_fraction: float,
        onset_fraction: float,
        offset_fraction: float,
    ) -> _Wave | None:
        if high - low < _samples(0.05):
            return None
        for scale in _WAVE_SCALES:
            qrs_modulus = float(
                np.max(
                    np.abs(
                        analysis.wavelet_coefficients[scale][
                            beat.qrs_onset : beat.qrs_offset + 1
                        ]
                    )
                )
            )
            wave = self._wave_at_scale(
                analysis,
                scale,
                low,
                high,
                amplitude_fraction * qrs_modulus,
                onset_fraction,
                offset_fraction,
            )
            if wave is not None:
                return wave
        return None

    def _wave_at_scale(
        self,
        analysis: _Analysis,
        scale: int,
        low: int,
        high: int,
        amplitude_threshold: float,
        onset_fraction: float,
        offset_fraction: float,
    ) -> _Wave | None:
        coefficients = analysis.wavelet_coefficients[scale]
        maxima = analysis.modulus_maxima[scale]
        lo = int(np.searchsorted(maxima, low, side="left"))
        hi = int(np.searchsorted(maxima, high, side="left"))
        significant = maxima[lo:hi]
        significant = significant[
            np.abs(coefficients[significant]) >= amplitude_threshold
        ]
        if len(significant) < 2:
            return None

        signs = np.sign(coefficients[significant])
        moduli = np.abs(coefficients[significant])
        pair_max = _samples(self.wave_pair_max_s)
        best, best_score = None, 0.0
        for i in range(len(significant) - 1):
            close = significant[i + 1] - significant[i] <= pair_max
            if (
                signs[i] != signs[i + 1]
                and close
                and moduli[i] + moduli[i + 1] > best_score
            ):
                best, best_score = i, moduli[i] + moduli[i + 1]
        if best is None:
            return None

        chain_gap = _samples(self.wave_chain_gap_s)
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
        crossings = _zero_crossings(
            coefficients,
            significant[best],
            significant[best + 1],
            falling=peak_is_maximum,
        )
        if not crossings:
            return None
        limit = _samples(self.wave_boundary_search_s)
        return _Wave(
            onset=_threshold_crossing(
                coefficients, int(significant[first]), -1, onset_fraction, limit
            ),
            peak=crossings[0][1],
            offset=_threshold_crossing(
                coefficients, int(significant[last]), +1, offset_fraction, limit
            ),
            peak_is_maximum=peak_is_maximum,
        )

    def _build_annotation(
        self, original: np.ndarray, beats: list[_Beat], rate_ratio: float
    ) -> ECGAnnotation:
        n = len(original)
        radius = max(1, int(self.peak_refine_s * rate_ratio * _WORKING_RATE))

        def boundary(position: float) -> int:
            return min(max(round(position * rate_ratio), 0), n - 1)

        def peak(position: float, is_maximum: bool) -> int:
            center = boundary(position)
            low, high = max(0, center - radius), min(n, center + radius + 1)
            segment = original[low:high]
            return low + int(np.argmax(segment) if is_maximum else np.argmin(segment))

        found: dict[Fiducials, list[int]] = {
            f: [] for f in Fiducials if f in self.supported_points
        }
        for beat in beats:
            found[Fiducials.QRS_ONSET].append(boundary(beat.qrs_onset))
            found[Fiducials.QRS_OFFSET].append(boundary(beat.qrs_offset))
            found[Fiducials.R_PEAK].append(peak(beat.r, beat.r_is_maximum))
            if beat.q is not None:
                found[Fiducials.Q_PEAK].append(peak(beat.q, not beat.r_is_maximum))
            if beat.s is not None:
                found[Fiducials.S_PEAK].append(peak(beat.s, not beat.r_is_maximum))
            for wave, onset, top, offset in (
                (beat.t, Fiducials.T_ONSET, Fiducials.T_PEAK, Fiducials.T_OFFSET),
                (beat.p, Fiducials.P_ONSET, Fiducials.P_PEAK, Fiducials.P_OFFSET),
            ):
                if wave is None:
                    continue
                found[top].append(peak(wave.peak, wave.peak_is_maximum))
                if wave.onset < wave.peak:
                    found[onset].append(boundary(wave.onset))
                if wave.offset > wave.peak:
                    found[offset].append(boundary(wave.offset))

        return ECGAnnotation.from_points(
            {
                fiducial: np.asarray(samples, dtype=np.int64)
                for fiducial, samples in found.items()
            }
        )