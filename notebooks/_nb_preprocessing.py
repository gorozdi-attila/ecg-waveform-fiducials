import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_waveform.core import ECGSignal
from ecg_waveform.visualization import plot_signal


def plot_preprocessing_steps(
    stages: list[tuple[str, ECGSignal]],
    start_sec: float = 0,
    interval_sec: float = 10,
    show_annotation: bool = True,
    figsize_per_row: float = 3,
) -> tuple[plt.Figure, list[plt.Axes]]:
    n_stages = len(stages)
    fig, axes = plt.subplots(n_stages, 1, figsize=(12, figsize_per_row * n_stages))
    axes = np.atleast_1d(axes)

    for ax, (name, signal) in zip(axes, stages):
        plot_signal(
            signal,
            start_sec=start_sec,
            interval_sec=interval_sec,
            show_annotation=show_annotation,
            ax=ax,
        )
        ax.set_title(name)

    plt.tight_layout()

    return fig, list(axes)


def compare(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    comparison = pd.concat(
        [raw, clean],
        axis=1,
        keys=["Raw", "Clean"],
    )

    return comparison