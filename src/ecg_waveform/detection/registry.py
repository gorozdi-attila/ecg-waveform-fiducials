from typing import Any

from .base import BaseDetector

DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_detector(name: str, cls: type[BaseDetector]) -> None:
    if name in DETECTOR_REGISTRY:
        raise ValueError(f"A detector named '{name}' is already registered.")
    DETECTOR_REGISTRY[name] = cls


def build_detector(name: str, params: dict[str, Any] | None = None) -> BaseDetector:
    if name not in DETECTOR_REGISTRY:
        raise KeyError(
            f"Unknown detector '{name}'. Available: {sorted(DETECTOR_REGISTRY)}"
        )
    return DETECTOR_REGISTRY[name](**(params or {}))
