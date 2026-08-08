from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from ecg_waveform.core import ECGSignal

from .registry import QUALITY_METRIC_REGISTRY


class Direction(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_RANGE = "target_range"


class QualityLevel(IntEnum):
    BAD = 0
    ACCEPTABLE = 1
    GOOD = 2


@dataclass(frozen=True)
class QualityMetricSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    direction: Direction = Direction.HIGHER_IS_BETTER

    good_threshold: float | tuple[float, float] | None = None
    acceptable_threshold: float | tuple[float, float] | None = None

    weight: float = 1.0

    def classify(self, value: float) -> QualityLevel | None:
        if value != value:  # NaN -> nem értékelhető
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

    @property
    def overall_level(self) -> QualityLevel:
        evaluated = [lvl for lvl in self.levels.values() if lvl is not None]
        return min(evaluated) if evaluated else QualityLevel.BAD

    @property
    def is_acceptable(self) -> bool:
        return self.overall_level >= QualityLevel.ACCEPTABLE

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(self.values)
        for name, level in self.levels.items():
            d[f"{name}_level"] = level.name if level is not None else None
        d["overall_level"] = self.overall_level.name
        return d

    def __repr__(self) -> str:
        lines = [f"QualityReport[{self.overall_level.name}]"]
        for name, value in self.values.items():
            level = self.levels[name]
            mark = level.name if level is not None else "--"
            lines.append(f"  {name:<22} = {value:>8.4f}  [{mark}]")
        return "\n".join(lines)


class SignalQualityAssessor:
    def __init__(self, specs: list[QualityMetricSpec]) -> None:
        self.specs = specs

    def __repr__(self) -> str:
        return f"SignalQualityAssessor(metrics={[s.name for s in self.specs]})"

    def assess(self, signal: ECGSignal) -> QualityReport:
        values: dict[str, float] = {}
        levels: dict[str, QualityLevel | None] = {}

        for spec in self.specs:
            if spec.name not in QUALITY_METRIC_REGISTRY:
                raise KeyError(
                    f"Unknown quality metric '{spec.name}'. "
                    f"Available: {sorted(QUALITY_METRIC_REGISTRY)}"
                )
            func = QUALITY_METRIC_REGISTRY[spec.name]
            value = func(signal, **spec.params)
            values[spec.name] = value
            levels[spec.name] = spec.classify(value)

        return QualityReport(values=values, levels=levels)

    @staticmethod
    def _parse_threshold(
        raw: float | list[float] | None,
    ) -> float | tuple[float, float] | None:
        if isinstance(raw, list):
            return tuple(raw)
        return raw

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SignalQualityAssessor":
        specs = [
            QualityMetricSpec(
                name=m["name"],
                params=m.get("params", {}),
                direction=Direction(m.get("direction", "higher_is_better")),
                good_threshold=cls._parse_threshold(m.get("good_threshold")),
                acceptable_threshold=cls._parse_threshold(
                    m.get("acceptable_threshold")
                ),
                weight=m.get("weight", 1.0),
            )
            for m in config["metrics"]
        ]
        return cls(specs)