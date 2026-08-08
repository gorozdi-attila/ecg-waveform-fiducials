from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_DIR


def load_yaml(filename: str) -> dict[str, Any]:
    path: Path = CONFIGS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")

    return data


def save_yaml(filename: str, data: dict[str, Any]) -> None:
    path: Path = CONFIGS_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )