from typing import Any

from ecg_waveform.core import ECGSignal

from .base import Direction
from .registry import build_quality_metric
from .report import QualityCriterion, QualityReport


class SignalQualityAssessor:
    def __init__(self, criteria: list[QualityCriterion]) -> None:
        self.criteria = criteria
        self._metrics = [
            build_quality_metric(c.metric_name, c.metric_params) for c in criteria
        ]

    def __repr__(self) -> str:
        return f"SignalQualityAssessor(criteria={[c.key for c in self.criteria]})"

    def assess(self, signal: ECGSignal) -> QualityReport:
        values: dict[str, float] = {}
        levels: dict[str, Any] = {}
        weights: dict[str, float] = {}

        for criterion, metric in zip(self.criteria, self._metrics, strict=True):
            value = metric.compute(signal)
            values[criterion.key] = value
            levels[criterion.key] = criterion.classify(value)
            weights[criterion.key] = criterion.weight

        return QualityReport(values=values, levels=levels, weights=weights)

    @staticmethod
    def _parse_threshold(
        raw: float | list[float] | None,
    ) -> float | tuple[float, float] | None:
        if isinstance(raw, list):
            return tuple(raw)
        return raw

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SignalQualityAssessor":
        criteria = [
            QualityCriterion(
                metric_name=m["name"],
                metric_params=m.get("params", {}),
                label=m.get("label"),
                direction=Direction(m.get("direction", "higher_is_better")),
                good_threshold=cls._parse_threshold(m.get("good_threshold")),
                acceptable_threshold=cls._parse_threshold(
                    m.get("acceptable_threshold")
                ),
                weight=m.get("weight", 1.0),
            )
            for m in config["metrics"]
        ]
        return cls(criteria)
