from typing import Any

from ecg_waveform.core import Registry

from .base import SignalQualityMetric

_METRICS: Registry[SignalQualityMetric] = Registry("quality metric")

QUALITY_METRIC_REGISTRY = _METRICS._items
register_quality_metric = _METRICS.register


def build_quality_metric(
    name: str, params: dict[str, Any] | None = None
) -> SignalQualityMetric:
    return _METRICS.build(name, params)
