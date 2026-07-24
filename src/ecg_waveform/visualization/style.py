import matplotlib.pyplot as plt

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
    "grid.linewidth": 0.3,
    "grid.color": "black",
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.loc": "upper right",
    "lines.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
}


def set_style(rc_params: dict | None = None) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(rc_params or DEFAULT_RCPARAMS)
