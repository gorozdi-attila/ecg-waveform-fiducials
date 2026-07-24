from pathlib import Path
from typing import Callable, Any

from collections import Counter

import numpy as np
import pandas as pd

from ecg_waveform.data import WFDBLoader
from ecg_waveform.core import ECGRecord, ECGSignal, ECGAnnotation

from ecg_waveform.quality import (
    kurtosis_sqi,
    qrs_power_sqi,
    powerline_noise_ratio,
    baseline_wander_ratio,
    flatline_ratio,
)


DEFAULT_OUTLIER_LIMIT_MV = 5.0
POWERLINE_FREQS_HZ = (50.0, 60.0)
SUMMARY_PRECISION = 3


AnnotationExtension = (
    str | dict[int | str, str] | Callable[[int, str | None], str | None]
)


def _round_value(value: Any, precision: int = SUMMARY_PRECISION) -> Any:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan

        return round(float(value), precision)

    return value


def _build_summary(items: dict[str, Any]) -> pd.DataFrame:
    rounded = {key: _round_value(value) for key, value in items.items()}

    return pd.DataFrame(
        {
            "Property": list(rounded.keys()),
            "Value": list(rounded.values()),
        }
    )


def _require_nonempty(record_names: list[str]) -> None:
    if not record_names:
        raise ValueError("record_names must contain at least one record name.")


def _format_annotation_extension(extension: AnnotationExtension) -> str:
    match extension:
        case None:
            return "None"
        case str() as ext:
            return ext
        case dict() as mapping:
            return ", ".join(
                f"{key}-{value}"
                for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
            )
        case _ if callable(extension):
            return getattr(extension, "__name__", extension.__class__.__name__)
        case _:
            return str(extension)


def _signal_stats(
    signal: ECGSignal,
    outlier_limit_mv: float = DEFAULT_OUTLIER_LIMIT_MV,
    powerline_freqs: tuple[float, ...] = POWERLINE_FREQS_HZ,
) -> dict[str, Any]:

    samples = np.asarray(signal.sample, dtype=float)

    nan_mask = np.isnan(samples)
    valid = samples[~nan_mask]

    stats: dict[str, Any] = {
        "nan_count": int(nan_mask.sum()),
        "nan_ratio": (100 * nan_mask.mean() if samples.size else np.nan),
        "outlier_ratio": (
            100 * np.mean(np.abs(valid) > outlier_limit_mv) if valid.size else np.nan
        ),
        "min_mv": np.min(valid) if valid.size else np.nan,
        "max_mv": np.max(valid) if valid.size else np.nan,
        "mean_mv": np.mean(valid) if valid.size else np.nan,
        "std_mv": np.std(valid) if valid.size else np.nan,
        "kurtosis_sqi": kurtosis_sqi(signal),
        "qrs_power_sqi": qrs_power_sqi(signal),
        "baseline_wander": baseline_wander_ratio(signal),
        "flatline": flatline_ratio(signal),
    }

    stats["dynamic_range"] = stats["max_mv"] - stats["min_mv"] if valid.size else np.nan

    for freq in powerline_freqs:
        stats[f"powerline_{freq}hz"] = powerline_noise_ratio(
            signal,
            powerline_freq=freq,
        )

    return stats


def summarize_record(record: ECGRecord) -> pd.DataFrame:

    return _build_summary(
        {
            "Record name": record.record_name,
            "Number of leads": len(record),
            "Leads": ", ".join(record.lead_names),
            "Sample rate (Hz)": record.sample_rate,
            "Number of samples": record.n_samples,
            "Duration (s)": (
                record.n_samples / record.sample_rate if record.sample_rate else np.nan
            ),
        }
    )


def summarize_signal(
    signal: ECGSignal,
    outlier_limit_mv: float = DEFAULT_OUTLIER_LIMIT_MV,
) -> pd.DataFrame:
    stats = _signal_stats(
        signal,
        outlier_limit_mv=outlier_limit_mv,
    )

    return _build_summary(
        {
            "Lead name": signal.lead_name,
            "Sample rate (Hz)": signal.sample_rate,
            "Number of samples": signal.n_samples,
            "Duration (s)": (
                signal.n_samples / signal.sample_rate if signal.sample_rate else np.nan
            ),
            "Min (mV)": stats["min_mv"],
            "Max (mV)": stats["max_mv"],
            "Mean (mV)": stats["mean_mv"],
            "Std (mV)": stats["std_mv"],
            "Dynamic range": stats["dynamic_range"],
            "Outlier ratio (%)": stats["outlier_ratio"],
            "NaN Samples": stats["nan_count"],
            "NaN ratio (%)": stats["nan_ratio"],
        }
    )


def summarize_annotation(
    annotation: ECGAnnotation,
    sample_rate: int,
) -> pd.DataFrame:
    unique_symbols = sorted(set(annotation.symbol))

    first_sample = min(annotation.sample)
    last_sample = max(annotation.sample)

    return _build_summary(
        {
            "Number of annotations": len(annotation.sample),
            "Number of unique symbols": len(unique_symbols),
            "Unique annotation symbols": ", ".join(unique_symbols),
            "First annotation sample": first_sample,
            "Last annotation sample": last_sample,
            "First annotation (s)": first_sample / sample_rate,
            "Last annotation (s)": last_sample / sample_rate,
        }
    )


def summarize_signal_quality(
    signal: ECGSignal,
    powerline_freqs: tuple[float, ...] = POWERLINE_FREQS_HZ,
) -> pd.DataFrame:
    stats = _signal_stats(
        signal,
        powerline_freqs=powerline_freqs,
    )

    items = {
        "Kurtosis SQI": stats["kurtosis_sqi"],
        "QRS Power SQI": stats["qrs_power_sqi"],
    }

    for freq in powerline_freqs:
        items[f"Powerline Noise Ratio ({freq} Hz)"] = stats[f"powerline_{freq}hz"]

    items["Baseline Wander Ratio"] = stats["baseline_wander"]
    items["Flatline Ratio"] = stats["flatline"]

    return _build_summary(items)


def summarize_database(
    database_path: Path,
    record_names: list[str],
    outlier_limit_mv: float = DEFAULT_OUTLIER_LIMIT_MV,
    powerline_freqs: tuple[float, ...] = POWERLINE_FREQS_HZ,
    skip_errors: bool = False,
) -> pd.DataFrame:
    _require_nonempty(record_names)

    loader = WFDBLoader(database_path)

    lead_names: set[str] = set()
    sample_rates: set[int] = set()

    durations: list[float] = []
    per_signal_stats: list[dict[str, Any]] = []

    total_samples = 0
    total_lead_channels = 0
    failed_records: list[str] = []

    for record_name in record_names:
        try:
            record = loader[record_name]

        except Exception as exc:
            if skip_errors:
                failed_records.append(record_name)
                continue

            raise RuntimeError(f"Failed to load record {record_name!r}") from exc

        total_samples += record.n_samples
        total_lead_channels += len(record)

        durations.append(record.n_samples / record.sample_rate)

        sample_rates.add(record.sample_rate)
        lead_names.update(record.lead_names)

        for signal in record:
            per_signal_stats.append(
                _signal_stats(
                    signal,
                    outlier_limit_mv,
                    powerline_freqs,
                )
            )

    def mean_stat(name: str) -> float:
        values = [item[name] for item in per_signal_stats]

        return float(np.nanmean(values)) if values else np.nan

    items = {
        "Number of records": len(record_names) - len(failed_records),
        "Total lead-channels": total_lead_channels,
        "Unique lead names": ", ".join(sorted(lead_names)),
        "Sample rates (Hz)": ", ".join(map(str, sorted(sample_rates))),
        "Total samples": total_samples,
        "Total duration (s)": (np.sum(durations) if durations else np.nan),
        "Mean duration (s)": (np.mean(durations) if durations else np.nan),
        "Median duration (s)": (np.median(durations) if durations else np.nan),
        "Mean NaN ratio (%)": mean_stat("nan_ratio"),
        "Mean outlier ratio (%)": mean_stat("outlier_ratio"),
        "Mean Kurtosis SQI": mean_stat("kurtosis_sqi"),
        "Mean QRS Power SQI": mean_stat("qrs_power_sqi"),
    }

    for freq in powerline_freqs:
        items[f"Mean Powerline Ratio ({freq} Hz)"] = mean_stat(f"powerline_{freq}hz")

    items["Mean Baseline Wander"] = mean_stat("baseline_wander")

    items["Mean Flatline Ratio"] = mean_stat("flatline")

    if skip_errors:
        items["Failed records"] = (
            ", ".join(failed_records) if failed_records else "none"
        )

    return _build_summary(items)


def summarize_database_annotations(
    database_path: Path,
    record_names: list[str],
    annotation_extension: AnnotationExtension,
) -> pd.DataFrame:
    _require_nonempty(record_names)

    loader = WFDBLoader(database_path, annotation_extension)

    annotated_signals = 0
    total_annotations = 0
    annotation_counts: list[int] = []
    symbols: Counter[str] = Counter()

    for record_name in record_names:
        record = loader[record_name]

        for signal in record:
            annotation = signal.annotation
            if annotation is None:
                continue

            n_annotations = len(annotation.sample)

            annotated_signals += 1
            total_annotations += n_annotations
            annotation_counts.append(n_annotations)
            symbols.update(annotation.symbol)

    return _build_summary(
        {
            "Annotation extension": _format_annotation_extension(annotation_extension),
            "Records": len(record_names),
            "Annotated signals": annotated_signals,
            "Total annotations": total_annotations,
            "Mean annotations per signal": np.mean(annotation_counts),
            "Median annotations per signal": np.median(annotation_counts),
            "Minimum annotations per signal": min(annotation_counts),
            "Maximum annotations per signal": max(annotation_counts),
            "Unique symbols": len(symbols),
            "Symbols": ", ".join(sorted(symbols)),
        }
    )
