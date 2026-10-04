from dataclasses import dataclass, field
from typing import Any

from .base import Direction, QualityLevel


@dataclass(frozen=True)
class QualityCriterion:
    metric_name: str
    metric_params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    direction: Direction = Direction.HIGHER_IS_BETTER
    good_threshold: float | tuple[float, float] | None = None
    acceptable_threshold: float | tuple[float, float] | None = None

    weight: float = 1.0

    @property
    def key(self) -> str:
        return self.label or self.metric_name

    def classify(self, value: float) -> QualityLevel | None:
        if value != value:
            return None

        if self.good_threshold is None and self.acceptable_threshold is None:
            return None

        match self.direction:
            case Direction.HIGHER_IS_BETTER:
                if self.good_threshold is not None and value >= self.good_threshold:
                    return QualityLevel.GOOD
                if (
                    self.acceptable_threshold is not None
                    and value >= self.acceptable_threshold
                ):
                    return QualityLevel.ACCEPTABLE
                return QualityLevel.BAD

            case Direction.LOWER_IS_BETTER:
                if self.good_threshold is not None and value <= self.good_threshold:
                    return QualityLevel.GOOD
                if (
                    self.acceptable_threshold is not None
                    and value <= self.acceptable_threshold
                ):
                    return QualityLevel.ACCEPTABLE
                return QualityLevel.BAD

            case Direction.TARGET_RANGE:
                if self.good_threshold is not None:
                    low, high = self.good_threshold
                    if low <= value <= high:
                        return QualityLevel.GOOD
                if self.acceptable_threshold is not None:
                    low, high = self.acceptable_threshold
                    if low <= value <= high:
                        return QualityLevel.ACCEPTABLE
                return QualityLevel.BAD


@dataclass(frozen=True)
class QualityReport:
    values: dict[str, float]
    levels: dict[str, QualityLevel | None]
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def overall_level(self) -> QualityLevel:
        evaluated = [lvl for lvl in self.levels.values() if lvl is not None]
        return min(evaluated) if evaluated else QualityLevel.BAD

    @property
    def is_acceptable(self) -> bool:
        return self.overall_level >= QualityLevel.ACCEPTABLE

    @property
    def weighted_score(self) -> float:
        scored = [
            (level.value / QualityLevel.GOOD.value, self.weights.get(key, 1.0))
            for key, level in self.levels.items()
            if level is not None
        ]
        total_weight = sum(w for _, w in scored)
        if not scored or total_weight == 0:
            return float("nan")

        return sum(score * w for score, w in scored) / total_weight

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(self.values)
        for key, level in self.levels.items():
            d[f"{key}_level"] = level.name if level is not None else None
        d["overall_level"] = self.overall_level.name
        d["weighted_score"] = self.weighted_score
        return d

    def __repr__(self) -> str:
        lines = [
            f"QualityReport[{self.overall_level.name}, score={self.weighted_score:.2f}]"
        ]
        for key, value in self.values.items():
            level = self.levels[key]
            mark = level.name if level is not None else "--"
            lines.append(f"  {key:<22} = {value:>8.4f}  [{mark}]")
        return "\n".join(lines)
