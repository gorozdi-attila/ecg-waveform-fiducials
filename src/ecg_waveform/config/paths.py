from pathlib import Path


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate project root (pyproject.toml not found).")


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
