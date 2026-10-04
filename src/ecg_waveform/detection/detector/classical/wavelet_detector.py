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
    link_maxima,
    localize_peak,
    samples,
    significant_crossings,
    to_working_rate,
    zero_crossings,
)
from ..base_detector import BaseDetector
from ..detector_registry import register_detector


@dataclass(slots=True)
class _QRSCandidate:
    r: float
    is_maximum: bool
    strength: float


@register_detector("martinez_wavelet")
@dataclass(repr=False, slots=True)
class MartinezWaveletDetector(BaseDetector):
    rms_window_s: float = DEFAULT_RMS_WINDOW_S
    threshold_factors: tuple[float, ...] = DEFAULT_QRS_THRESHOLD_FACTORS

    pair_max_s: float = 0.15
    min_line_scales: int = 3
    refractory_period_s: float = 0.2

    searchback_gap_ratio: float = 1.5
    searchback_threshold_ratio: float = 0.5

    peak_refine_s: float = DEFAULT_PEAK_REFINE_S

    @property
    def supported_points(self) -> frozenset[Fiducials]:
        return frozenset({Fiducials.R_PEAK})

    def _link_pair(
        self,
        analysis: Analysis,
        first: int,
        second: int,
        thresholds: dict[int, np.ndarray],
    ) -> _QRSCandidate | None:
        line_a = link_maxima(analysis, 4, first, thresholds)
        line_b = link_maxima(analysis, 4, second, thresholds)
        if min(len(line_a), len(line_b)) < self.min_line_scales:
            return None
        if 2 not in line_a or 2 not in line_b or line_a[2] >= line_b[2]:
            return None

        w2 = analysis.wavelet_coefficients[2]
        is_maximum = bool(analysis.wavelet_coefficients[4][first] > 0)
        crossings = zero_crossings(w2, line_a[2], line_b[2], falling=is_maximum)
        if not crossings:
            return None

        return _QRSCandidate(
            r=crossings[0][1],
            is_maximum=is_maximum,
            strength=float(abs(w2[line_a[2]]) + abs(w2[line_b[2]])),
        )

    def _search(
        self, analysis: Analysis, threshold_ratio: float, start: int, stop: int
    ) -> list[_QRSCandidate]:
        coefficients = analysis.wavelet_coefficients
        thresholds = analysis.thresholds(self.threshold_factors, threshold_ratio)
        coarse = [
            int(p)
            for p in analysis.modulus_maxima[4]
            if start <= p < stop and abs(coefficients[4][p]) >= thresholds[4][p]
        ]

        pair_max = samples(self.pair_max_s)
        found: list[_QRSCandidate] = []
        i = 0
        while i < len(coarse) - 1:
            first, second = coarse[i], coarse[i + 1]
            opposite = np.sign(coefficients[4][first]) != np.sign(coefficients[4][second])

            candidate = None
            if opposite and second - first <= pair_max:
                candidate = self._link_pair(analysis, first, second, thresholds)

            if candidate is not None:
                found.append(candidate)
                i += 2
            else:
                i += 1
        return found

    def _merge_redundant(self, candidates: list[_QRSCandidate]) -> list[_QRSCandidate]:
        refractory = samples(self.refractory_period_s)
        kept: list[_QRSCandidate] = []
        for candidate in sorted(candidates, key=lambda c: c.r):
            if kept and candidate.r - kept[-1].r < refractory:
                if candidate.strength > kept[-1].strength:
                    kept[-1] = candidate
            else:
                kept.append(candidate)
        return kept

    def _detect_candidates(self, analysis: Analysis) -> list[_QRSCandidate]:
        candidates = self._merge_redundant(self._search(analysis, 1.0, 0, analysis.n_samples))

        if len(candidates) >= 2:
            median_rr = float(np.median(np.diff([c.r for c in candidates])))
            margin = samples(self.refractory_period_s)
            recovered: list[_QRSCandidate] = []
            for left, right in zip(candidates[:-1], candidates[1:]):
                if right.r - left.r > self.searchback_gap_ratio * median_rr:
                    recovered += self._search(
                        analysis,
                        self.searchback_threshold_ratio,
                        int(left.r) + margin,
                        int(right.r) - margin,
                    )
            candidates = self._merge_redundant(candidates + recovered)

        return candidates

    def _refine_at_scale_1(
        self,
        analysis: Analysis,
        thresholds: dict[int, np.ndarray],
        candidate: _QRSCandidate,
    ) -> float:
        center = int(round(candidate.r))
        fine = significant_crossings(
            analysis, thresholds, 1, center - 3, center + 3, falling=candidate.is_maximum
        )
        return min(fine, key=lambda c: abs(c - candidate.r)) if fine else candidate.r

    def detect(self, signal: ECGSignal) -> ECGAnnotation:
        original = np.asarray(signal.sample, dtype=np.float64)
        rate_ratio = signal.sample_rate / WORKING_RATE

        working = to_working_rate(original, signal.sample_rate)
        if len(working) < MIN_WORKING_SAMPLES:
            return ECGAnnotation.from_points({})

        analysis = Analysis.from_ecg(working, self.rms_window_s)
        thresholds = analysis.thresholds(self.threshold_factors)

        r_samples = [
            localize_peak(
                original,
                self._refine_at_scale_1(analysis, thresholds, candidate),
                candidate.is_maximum,
                rate_ratio,
                self.peak_refine_s,
            )
            for candidate in self._detect_candidates(analysis)
        ]

        return ECGAnnotation.from_points(
            {Fiducials.R_PEAK: np.unique(np.asarray(r_samples, dtype=np.int64))}
        )