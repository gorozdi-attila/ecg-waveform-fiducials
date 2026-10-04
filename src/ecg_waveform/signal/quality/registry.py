from collections.abc import Callable
from typing import Any

from .base import SignalQualityMetric

QUALITY_METRIC_REGISTRY: dict[str, type[SignalQualityMetric]] = {}


def register_quality_metric(
    name: str,
) -> Callable[[type[SignalQualityMetric]], type[SignalQualityMetric]]:
    def _register(cls: type[SignalQualityMetric]) -> type[SignalQualityMetric]:
        if name in QUALITY_METRIC_REGISTRY:
            raise ValueError(f"A quality metric named '{name}' is already registered.")
        QUALITY_METRIC_REGISTRY[name] = cls
        return cls

    return _register


def build_quality_metric(
    name: str, params: dict[str, Any] | None = None
) -> SignalQualityMetric:
    if name not in QUALITY_METRIC_REGISTRY:
        raise KeyError(
            f"Unknown quality metric '{name}'. Available: {sorted(QUALITY_METRIC_REGISTRY)}"
        )
    return QUALITY_METRIC_REGISTRY[name](**(params or {}))
