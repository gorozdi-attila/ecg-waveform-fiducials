from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from ecg_waveform.config import load_yaml
from ecg_waveform.core import ECGAnnotation
from ecg_waveform.data import WFDBLoader

FIDUCIAL_INTERVALS = load_yaml("fiducial_normal_ranges.yaml")
SUMMARY_PRECISION = 3


def _round_df(
    df: pd.DataFrame,
    precision: int = SUMMARY_PRECISION,
) -> pd.DataFrame:

    return df.round(precision)


def _require_nonempty(record_names: list[str]) -> None:
    if not record_names:
        raise ValueError("record_names must contain at least one record name.")


def _extract_waves(
    annotation: ECGAnnotation,
) -> list[dict[str, Any]]:

    labels = load_yaml("annotation_labels.yaml")

    type_map = {
        "p_peaks": "P",
        "r_peaks": "QRS",
        "t_peaks": "T",
    }

    wave_map = {
        symbol: type_map[group]
        for group, symbols in labels.items()
        for symbol in symbols
    }

    samples = annotation.sample
    symbols = annotation.symbol

    if len(samples) != len(symbols):
        raise ValueError("Annotation samples and symbols must have equal length.")

    waves = []

    for i in range(len(symbols) - 2):
        s0, s1, s2 = symbols[i : i + 3]
        t0, t1, t2 = samples[i : i + 3]

        if s0 == "(" and s2 == ")" and s1 in wave_map:
            waves.append(
                {
                    "type": wave_map[s1],
                    "onset": t0,
                    "peak": t1,
                    "offset": t2,
                }
            )

    return waves


def _empty_beat() -> dict[str, Any]:

    return {
        f"{prefix}_{field}": None
        for prefix in ("p", "qrs", "t")
        for field in ("onset", "peak", "offset")
    }


def _match_beats(
    waves: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    qrs = sorted(
        [w for w in waves if w["type"] == "QRS"],
        key=lambda x: x["peak"],
    )

    p_waves = sorted(
        [w for w in waves if w["type"] == "P"],
        key=lambda x: x["peak"],
    )

    t_waves = sorted(
        [w for w in waves if w["type"] == "T"],
        key=lambda x: x["peak"],
    )

    beats = []

    for index, qrs_wave in enumerate(qrs):
        beat = _empty_beat()

        previous_qrs = qrs[index - 1]["peak"] if index else -np.inf
        next_qrs = qrs[index + 1]["peak"] if index + 1 < len(qrs) else np.inf

        p_candidates = [
            p for p in p_waves if previous_qrs < p["peak"] < qrs_wave["peak"]
        ]

        if p_candidates:
            p = p_candidates[-1]

            if p["offset"] <= qrs_wave["onset"]:
                for k in ("onset", "peak", "offset"):
                    beat[f"p_{k}"] = p[k]

        t_candidates = [t for t in t_waves if qrs_wave["peak"] < t["peak"] < next_qrs]

        if t_candidates:
            t = t_candidates[0]

            if qrs_wave["offset"] <= t["onset"]:
                for k in ("onset", "peak", "offset"):
                    beat[f"t_{k}"] = t[k]

        for k in ("onset", "peak", "offset"):
            beat[f"qrs_{k}"] = qrs_wave[k]

        beats.append(beat)

    return beats


def compute_fiducials(
    annotation: ECGAnnotation,
    sample_rate: int,
) -> pd.DataFrame:

    beats = pd.DataFrame(_match_beats(_extract_waves(annotation)))

    if beats.empty:
        return pd.DataFrame()

    beats = pd.DataFrame(beats)
    summaries = pd.DataFrame(index=beats.index)

    ms_per_sample = 1000.0 / sample_rate

    summaries["RR interval"] = beats["qrs_peak"].diff() * ms_per_sample

    valid_p = beats["p_peak"].dropna()

    pp = valid_p.diff() * ms_per_sample

    summaries["PP interval"] = pp.reindex(beats.index)

    summaries["PR interval"] = (beats["qrs_onset"] - beats["p_onset"]) * ms_per_sample
    summaries["QT interval"] = (beats["t_offset"] - beats["qrs_onset"]) * ms_per_sample
    summaries["ST segment"] = (beats["t_onset"] - beats["qrs_offset"]) * ms_per_sample

    summaries["P duration"] = (beats["p_offset"] - beats["p_onset"]) * ms_per_sample
    summaries["QRS duration"] = (
        beats["qrs_offset"] - beats["qrs_onset"]
    ) * ms_per_sample
    summaries["T duration"] = (beats["t_offset"] - beats["t_onset"]) * ms_per_sample

    rr_sec = summaries["RR interval"] / 1000.0

    valid_rr = rr_sec > 0

    summaries["QTc Bazett"] = np.where(
        valid_rr, summaries["QT interval"] / np.sqrt(rr_sec), np.nan
    )
    summaries["QTc Fridericia"] = np.where(
        valid_rr, summaries["QT interval"] / np.cbrt(rr_sec), np.nan
    )

    return summaries


def summarize_database_fiducial(
    database_path: Path,
    record_names: list[str],
    annotation_extension: str,
    fiducial_intervals: dict = FIDUCIAL_INTERVALS,
    show_summary: bool = True,
) -> pd.DataFrame:

    _require_nonempty(record_names)

    summaries = []

    loader = WFDBLoader(
        database_path,
        annotation_extension,
    )

    for record_name in record_names:
        record = loader[record_name]

        annotation = record.get_signal(0).annotation

        beat_df = compute_fiducials(annotation, record.sample_rate)

        if beat_df is None or beat_df.dropna(how="all").empty:
            continue

        beat_df["record"] = record_name
        summaries.append(beat_df)

    if not summaries:
        return pd.DataFrame()

    summaries = [df.dropna(axis=1, how="all") for df in summaries]

    fiducial_summaries = pd.concat(summaries, ignore_index=True)

    if show_summary:
        print(f"Successfully matched beats: {len(fiducial_summaries)}")

        summary = pd.DataFrame(
            {
                "Mean (ms)": fiducial_summaries.mean(numeric_only=True),
                "Std (ms)": fiducial_summaries.std(numeric_only=True),
                "Min (ms)": fiducial_summaries.min(numeric_only=True),
                "Max (ms)": fiducial_summaries.max(numeric_only=True),
            }
        )

        summary["Normal range"] = summary.index.map(
            lambda x: fiducial_intervals.get(x, "")
        )

        display(_round_df(summary))

    return fiducial_summaries


def plot_fiducials_distributions(
    fiducials_summary: pd.DataFrame,
    normal_intervals: dict = FIDUCIAL_INTERVALS,
    ncols: int = 5,
) -> None:
    cols = [c for c in normal_intervals if c in fiducials_summary.columns]

    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 3.5 * nrows))

    axes = np.array(axes).flatten()

    for ax, col in zip(axes, cols):
        data = fiducials_summary[col].dropna()

        sns.histplot(
            data, bins="auto", kde=True, linewidth=0, ax=ax, label="distribution"
        )

        low, high = normal_intervals[col]
        ax.axvline(low, color="red", linestyle="--", label="Normal interval")
        ax.axvline(high, color="red", linestyle="--")

        ax.axvline(data.mean(), color="green", linestyle="--", label="Mean interval")

        ax.set_title(col)
        ax.set_xlabel("Time (ms)")

    for ax in axes[len(cols) :]:
        ax.set_visible(False)

    fig.suptitle("Fiducial intervals Distributions", fontweight="bold", fontsize="14")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=False,
    )

    plt.tight_layout()

    return fig, axes
