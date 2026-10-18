from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_DIR


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidate = CONFIGS_DIR / path
    if candidate.exists():
        return candidate
    return path if path.exists() else candidate


def load_yaml(filename: str | Path) -> dict[str, Any]:
    path = _resolve(filename)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")

    return data


def save_yaml(filename: str | Path, data: dict[str, Any]) -> None:
    path = Path(filename)
    
    if not path.is_absolute():
        path = CONFIGS_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
