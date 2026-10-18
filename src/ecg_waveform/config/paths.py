import os
from pathlib import Path

_ENV_ROOT = "ECG_WAVEFORM_ROOT"


def _project_root() -> Path:
    env = os.environ.get(_ENV_ROOT)
    
    if env:
        return Path(env).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent

    return Path.cwd()


PROJECT_ROOT: Path = _project_root()

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

DATA_DIR: Path = PROJECT_ROOT / "data"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABELS_DIR: Path = RESULTS_DIR / "tabels"


DIRECTORIES: tuple[Path] = (
    CONFIGS_DIR,
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    TABELS_DIR,
)


def ensure_directories() -> None:
    for path in DIRECTORIES:
        path.mkdir(parents=True, exist_ok=True)
