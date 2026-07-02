from functools import cache
from pathlib import Path

import wfdb

def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate project root (pyproject.toml not found).")


PROJECT_ROOT: Path = _project_root()

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"

MITBIH_DIR: Path = DATA_DIR / "mitbih"
LUDB_DIR: Path = DATA_DIR / "ludb"

RESULTS_DIR: Path = PROJECT_ROOT / "results"


DIRECTORIES: tuple[Path] = (
    CONFIGS_DIR,
    DATA_DIR,
    MITBIH_DIR,
    LUDB_DIR,
    RESULTS_DIR,
)


def ensure_directories() -> None:
    for path in DIRECTORIES:
        path.mkdir(parents=True, exist_ok=True)


@cache
def get_mitbih_record_names() -> list[str]:
    return wfdb.get_record_list("mitdb")


@cache
def get_ludb_record_names() -> list[str]:
    return wfdb.get_record_list("ludb")