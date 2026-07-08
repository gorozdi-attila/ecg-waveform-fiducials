import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RCPARAMS: dict = {
    "figure.figsize": (12, 4),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "normal",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "grid.color": "#808080",
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.loc": "lower left",
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

def set_style(rc_params: dict | None = None) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(rc_params or DEFAULT_RCPARAMS)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)