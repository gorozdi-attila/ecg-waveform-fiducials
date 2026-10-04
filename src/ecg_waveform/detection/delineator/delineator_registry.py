from collections.abc import Callable
from typing import Any

from .base_delineator import BaseDelineator

DELINEATOR_REGISTRY: dict[str, type[BaseDelineator]] = {}


def register_delineator(name: str) -> Callable[[type[BaseDelineator]], type[BaseDelineator]]:
    def _register(cls: type[BaseDelineator]) -> type[BaseDelineator]:
        if name in DELINEATOR_REGISTRY:
            raise ValueError(f"A delineator named '{name}' is already registered.")
        DELINEATOR_REGISTRY[name] = cls
        return cls

    return _register


def build_delineator(name: str, params: dict[str, Any] | None = None) -> BaseDelineator:
    if name not in DELINEATOR_REGISTRY:
        raise KeyError(
            f"Unknown delineator '{name}'. Available: {sorted(DELINEATOR_REGISTRY)}"
        )
    return DELINEATOR_REGISTRY[name](**(params or {}))


def available_delineators() -> list[str]:
    return list(sorted(DELINEATOR_REGISTRY))
