from dataclasses import dataclass

import numpy as np

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials
from ecg_waveform.core.entities import _as_symbols


@dataclass(frozen=True, eq=False)
class MatchResult:
    matched_reference_idx: np.ndarray
    matched_predicted_idx: np.ndarray
    n_reference: int
    n_predicted: int

    @property
    def true_positives(self) -> int:
        return len(self.matched_reference_idx)

    @property
    def false_negatives(self) -> int:
        return self.n_reference - self.true_positives

    @property
    def false_positives(self) -> int:
        return self.n_predicted - self.true_positives


@dataclass(frozen=True)
class EvaluationMetrics:
    symbol: str
    n_reference: int
    n_predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_absolute_timing_error_ms: float | None
    std_absolute_timing_error_ms: float | None
    mean_timing_error_ms: float | None = None
    std_timing_error_ms: float | None = None

    @property
    def sensitivity(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else float("nan")

    @property
    def ppv(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else float("nan")

    @property
    def f1(self) -> float:
        se, ppv = self.sensitivity, self.ppv
        if np.isnan(se) or np.isnan(ppv) or (se + ppv) == 0:
            return float("nan")
        return 2 * se * ppv / (se + ppv)

    @property
    def detection_error_rate(self) -> float:
        if self.n_reference == 0:
            return float("nan")
        return (self.false_positives + self.false_negatives) / self.n_reference

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "Symbol": self.symbol,
            "Reference": self.n_reference,
            "Predicted": self.n_predicted,
            "TP": self.true_positives,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "Sensitivity": self.sensitivity,
            "Positive Predictive Value": self.ppv,
            "F1 Score": self.f1,
            "Detection Error Rate": self.detection_error_rate,
            "Mean Absolute Timing Error": self.mean_absolute_timing_error_ms,
            "Mean Absolute Timing Error Std": self.std_absolute_timing_error_ms,
            "Mean Timing Error": self.mean_timing_error_ms,
            "Timing Error Std": self.std_timing_error_ms,
        }


def match_points(
    reference: np.ndarray,
    predicted: np.ndarray,
    tolerance_samples: int,
) -> MatchResult:
    if tolerance_samples < 0:
        raise ValueError(
            f"tolerance_samples must be non-negative, got {tolerance_samples}"
        )

    reference = np.asarray(reference, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)

    if len(reference) == 0 or len(predicted) == 0:
        return MatchResult(empty, empty, len(reference), len(predicted))

    lo = np.searchsorted(predicted, reference - tolerance_samples, side="left")
    hi = np.searchsorted(predicted, reference + tolerance_samples, side="right")
    counts = hi - lo
    if counts.sum() == 0:
        return MatchResult(empty, empty, len(reference), len(predicted))

    ref_idx = np.repeat(np.arange(len(reference)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    pred_idx = np.repeat(lo, counts) + offsets
    dist = np.abs(predicted[pred_idx] - reference[ref_idx])

    order = np.lexsort((pred_idx, ref_idx, dist))
    used_ref = np.zeros(len(reference), dtype=bool)
    used_pred = np.zeros(len(predicted), dtype=bool)
    matched_ref: list[int] = []
    matched_pred: list[int] = []

    for k in order:
        i, j = int(ref_idx[k]), int(pred_idx[k])
        if used_ref[i] or used_pred[j]:
            continue
        used_ref[i] = used_pred[j] = True
        matched_ref.append(i)
        matched_pred.append(j)

    matched_ref_arr = np.array(matched_ref, dtype=np.int64)
    matched_pred_arr = np.array(matched_pred, dtype=np.int64)
    sort = np.argsort(matched_ref_arr)

    return MatchResult(
        matched_reference_idx=matched_ref_arr[sort],
        matched_predicted_idx=matched_pred_arr[sort],
        n_reference=len(reference),
        n_predicted=len(predicted),
    )


def compute_metrics(
    signal: ECGSignal,
    reference: ECGAnnotation,
    predicted: ECGAnnotation,
    symbol: Fiducials | str,
    tolerance_ms: float = 50.0,
) -> EvaluationMetrics:
    (symbol_value,) = _as_symbols(symbol)
    if tolerance_ms < 0:
        raise ValueError(f"tolerance_ms must be non-negative, got {tolerance_ms}")

    reference = reference.filter(symbol_value)
    predicted = predicted.filter(symbol_value)

    tolerance_samples = round(tolerance_ms / 1000 * signal.sample_rate)

    match = match_points(
        reference.sample,
        predicted.sample,
        tolerance_samples,
    )

    signed_error_ms: np.ndarray | None = None
    if match.true_positives > 0:
        ref_matched = reference.sample[match.matched_reference_idx]
        pred_matched = predicted.sample[match.matched_predicted_idx]
        signed_error_ms = (
            (pred_matched.astype(np.float64) - ref_matched.astype(np.float64))
            / signal.sample_rate
            * 1000
        )
    timing_error_ms = None if signed_error_ms is None else np.abs(signed_error_ms)

    return EvaluationMetrics(
        symbol=symbol_value,
        n_reference=match.n_reference,
        n_predicted=match.n_predicted,
        true_positives=match.true_positives,
        false_positives=match.false_positives,
        false_negatives=match.false_negatives,
        mean_absolute_timing_error_ms=(
            float(np.mean(timing_error_ms)) if timing_error_ms is not None else None
        ),
        std_absolute_timing_error_ms=(
            float(np.std(timing_error_ms)) if timing_error_ms is not None else None
        ),
        mean_timing_error_ms=(
            float(np.mean(signed_error_ms)) if signed_error_ms is not None else None
        ),
        std_timing_error_ms=(
            float(np.std(signed_error_ms)) if signed_error_ms is not None else None
        ),
    )
