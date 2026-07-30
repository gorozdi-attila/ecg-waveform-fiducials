from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import wfdb

from ecg_waveform.core import ECGAnnotation, ECGRecord, ECGSignal


class BaseDataLoader(ABC):
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)

    @lru_cache(maxsize=None)
    @abstractmethod
    def __getitem__(self, record_name: str) -> ECGRecord: ...


AnnotationExtension = (
    str | dict[int | str, str] | Callable[[int, str | None], str | None] | None
)


class WFDBLoader(BaseDataLoader):
    def __init__(
        self, dataset_root: Path, annotation_extension: AnnotationExtension = None
    ):
        super().__init__(dataset_root)
        self.annotation_extension = annotation_extension

    def _resolve_extension(self, channel: int, lead_name: str | None) -> str | None:
        match self.annotation_extension:
            case None:
                return None
            case str() as ext:
                return ext
            case dict() as mapping:
                if lead_name is not None and lead_name in mapping:
                    return mapping[lead_name]
                return mapping.get(channel)
            case _ if callable(self.annotation_extension):
                return self.annotation_extension(channel, lead_name)
            case _:
                raise TypeError(
                    f"Invalid annotation_extension type: {type(self.annotation_extension)!r}"
                )

    def _load_annotation(
        self, record_path: Path, extension: str, channel: int
    ) -> ECGAnnotation:
        try:
            annotation = wfdb.rdann(str(record_path), extension)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load annotation '{extension}' for '{record_path}'."
            ) from e

        symbol = np.asarray(annotation.symbol)
        sample = np.asarray(annotation.sample, dtype=np.int64)

        chan = getattr(annotation, "chan", None)
        if chan is not None:
            chan_arr = np.asarray(chan)

            if chan_arr.size and np.unique(chan_arr).size > 1:
                mask = chan_arr == channel
                symbol = symbol[mask]
                sample = sample[mask]

        return ECGAnnotation(symbol=symbol, sample=sample)

    @lru_cache(maxsize=None)
    def __getitem__(self, record_name: str) -> ECGRecord:
        record_path = self.dataset_root / record_name
        record = wfdb.rdrecord(str(record_path))

        signals = []
        for channel in range(record.n_sig):
            lead_name = record.sig_name[channel]
            extension = self._resolve_extension(channel, lead_name)

            annotation = None
            if extension is not None:
                annotation = self._load_annotation(record_path, extension, channel)

            signal = ECGSignal(
                sample=np.asarray(record.p_signal[:, channel]),
                sample_rate=int(record.fs),
                channel=channel,
                lead_name=lead_name,
                annotation=annotation,
            )
            signals.append(signal)

        return ECGRecord(
            record_name=record_name,
            signals=signals,
        )
