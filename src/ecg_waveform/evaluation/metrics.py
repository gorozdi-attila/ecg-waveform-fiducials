from dataclasses import dataclass

import numpy as np

from ecg_waveform.core import ECGSignal, ECGAnnotation, Fiducials


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
    symbol: Fiducials
    n_reference: int
    n_predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_absolute_timing_error_ms: float | None
    std_absolute_timing_error_ms: float | None

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
        }


def match_points(
    reference: np.ndarray,
    predicted: np.ndarray,
    tolerance_samples: int,
) -> MatchResult:
    i, j = 0, 0
    matched_ref: list[int] = []
    matched_pred: list[int] = []

    while i < len(reference) and j < len(predicted):
        diff = int(predicted[j]) - int(reference[i])

        if diff < -tolerance_samples:
            j += 1
        elif diff > tolerance_samples:
            i += 1
        else:
            matched_ref.append(i)
            matched_pred.append(j)
            i += 1
            j += 1

    return MatchResult(
        matched_reference_idx=np.array(matched_ref, dtype=np.int64),
        matched_predicted_idx=np.array(matched_pred, dtype=np.int64),
        n_reference=len(reference),
        n_predicted=len(predicted),
    )


def compute_metrics(
    signal: ECGSignal,
    reference: ECGAnnotation,
    predicted: ECGAnnotation,
    symbol: Fiducials,
    tolerance_ms: float = 50.0,
) -> EvaluationMetrics:
    reference = reference.filter(symbol)
    predicted = predicted.filter(symbol)

    tolerance_samples = int(round(tolerance_ms / 1000 * signal.sample_rate))

    match = match_points(
        reference.sample,
        predicted.sample,
        tolerance_samples,
    )

    timing_error_ms: np.ndarray | None = None
    if match.true_positives > 0:
        ref_matched = reference.sample[match.matched_reference_idx]
        pred_matched = predicted.sample[match.matched_predicted_idx]
        timing_error_ms = (
            np.abs(pred_matched.astype(np.float64) - ref_matched.astype(np.float64))
            / signal.sample_rate
            * 1000
        )

    return EvaluationMetrics(
        symbol=symbol,
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
    )
