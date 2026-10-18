import importlib
import importlib.util
import pkgutil
from typing import Any

from ecg_waveform.core import Registry

from .base_detector import BasePQRSTDelineator, BasePQRSTDetector

DETECTORS: Registry[BasePQRSTDetector] = Registry("detector")
DELINEATORS: Registry[BasePQRSTDelineator] = Registry("delineator")

register_detector = DETECTORS.register
register_delineator = DELINEATORS.register

_ALGORITHM_PACKAGES = (
    "ecg_waveform.detection.classical",
    "ecg_waveform.detection.deep",
)


def build_detector(name: str, params: dict[str, Any] | None = None) -> BasePQRSTDetector:
    return DETECTORS.build(name, params)


def build_delineator(name: str, params: dict[str, Any] | None = None) -> BasePQRSTDelineator:
    return DELINEATORS.build(name, params)


def available_detectors() -> list[str]:
    return DETECTORS.names()


def available_delineators() -> list[str]:
    return DELINEATORS.names()

def discover() -> None:
    for package_name in _ALGORITHM_PACKAGES:
        if importlib.util.find_spec(package_name) is None:
            continue
        package = importlib.import_module(package_name)
        for module in pkgutil.walk_packages(
            package.__path__, prefix=f"{package_name}."
        ):
            importlib.import_module(module.name)
