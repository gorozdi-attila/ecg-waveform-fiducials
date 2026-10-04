from collections.abc import Callable
from typing import Any

from .base_detector import BaseDetector

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_detector(name: str) -> Callable[[type[BaseDetector]], type[BaseDetector]]:
    def _register(cls: type[BaseDetector]) -> type[BaseDetector]:
        if name in DETECTOR_REGISTRY:
            raise ValueError(f"A detector named '{name}' is already registered.")
        DETECTOR_REGISTRY[name] = cls
        return cls

    return _register


def build_detector(name: str, params: dict[str, Any] | None = None) -> BaseDetector:
    if name not in DETECTOR_REGISTRY:
        raise KeyError(
            f"Unknown detector '{name}'. Available: {sorted(DETECTOR_REGISTRY)}"
        )
    return DETECTOR_REGISTRY[name](**(params or {}))


def available_detectors() -> list[str]:
    return list(sorted(DETECTOR_REGISTRY))
