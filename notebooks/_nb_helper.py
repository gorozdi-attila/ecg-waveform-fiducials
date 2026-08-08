from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from ecg_waveform.config import FIGURES_DIR, ensure_directories
from ecg_waveform.visualization import set_style


def setup_notebook() -> None:
    ensure_directories()

    set_style()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)


def save_plot(fig: Figure, filename: str) -> Path:
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight")

    return path


def compare_tables(tables: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    if len(tables) != len(names):
        raise ValueError("`tables` and `names` must have the same length.")

    comparison = pd.concat(
        tables,
        axis=1,
        keys=names,
    )

    return comparison
