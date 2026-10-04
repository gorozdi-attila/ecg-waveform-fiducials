from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ecg_waveform.config import load_yaml
from ecg_waveform.core import ECGAnnotation, Fiducials


class GenericMarker(str, Enum):
    ONSET = "generic_onset"
    OFFSET = "generic_offset"


GROUP_TO_FIDUCIAL: dict[str, Fiducials | GenericMarker] = {
    "p_peaks": Fiducials.P_PEAK,
    "q_peaks": Fiducials.Q_PEAK,
    "r_peaks": Fiducials.R_PEAK,
    "s_peaks": Fiducials.S_PEAK,
    "t_peaks": Fiducials.T_PEAK,
    "wave_onset": GenericMarker.ONSET,
    "wave_offset": GenericMarker.OFFSET,
}

PEAK_TO_WAVE: dict[Fiducials, str] = {
    Fiducials.P_PEAK: "P",
    Fiducials.Q_PEAK: "QRS",
    Fiducials.R_PEAK: "QRS",
    Fiducials.S_PEAK: "QRS",
    Fiducials.T_PEAK: "T",
}

WAVE_ONSET: dict[str, Fiducials] = {
    "P": Fiducials.P_ONSET,
    "QRS": Fiducials.QRS_ONSET,
    "T": Fiducials.T_ONSET,
}

WAVE_OFFSET: dict[str, Fiducials] = {
    "P": Fiducials.P_OFFSET,
    "QRS": Fiducials.QRS_OFFSET,
    "T": Fiducials.T_OFFSET,
}


@dataclass(frozen=True, slots=True)
class FiducialMapper:
    symbol_to_fiducial: dict[str, Fiducials | GenericMarker] = field(
        default_factory=dict
    )

    def __repr__(self) -> str:
        per_fiducial: dict[str, int] = {}
        for fiducial in self.symbol_to_fiducial.values():
            per_fiducial[fiducial.value] = per_fiducial.get(fiducial.value, 0) + 1

        return (
            f"AnnotationSymbolMapper(n_symbols={len(self.symbol_to_fiducial)}, "
            f"per_fiducial={per_fiducial})"
        )

    def __len__(self) -> int:
        return len(self.symbol_to_fiducial)

    @staticmethod
    def _nearest_wave(
        resolved: list[Fiducials | None], idx: int, direction: int
    ) -> str | None:
        i = idx + direction
        n = len(resolved)
        while 0 <= i < n:
            fiducial = resolved[i]
            if fiducial is not None:
                return PEAK_TO_WAVE.get(fiducial)
            i += direction
        return None

    def translate(self, annotation: ECGAnnotation) -> ECGAnnotation:
        if len(annotation) == 0:
            return annotation

        raw_symbols = [str(s) for s in annotation.symbol]
        n = len(raw_symbols)
        resolved: list[Fiducials | None] = [None] * n

        for idx, raw_symbol in enumerate(raw_symbols):
            mapped = self.symbol_to_fiducial.get(raw_symbol)
            if isinstance(mapped, Fiducials):
                resolved[idx] = mapped

        for idx, raw_symbol in enumerate(raw_symbols):
            marker = self.symbol_to_fiducial.get(raw_symbol)

            if marker is GenericMarker.ONSET:
                wave = self._nearest_wave(resolved, idx, direction=1)
                if wave is not None:
                    resolved[idx] = WAVE_ONSET[wave]
            elif marker is GenericMarker.OFFSET:
                wave = self._nearest_wave(resolved, idx, direction=-1)
                if wave is not None:
                    resolved[idx] = WAVE_OFFSET[wave]

        translated_symbols: list[str] = []
        kept_indices: list[int] = []
        for idx, fiducial in enumerate(resolved):
            if fiducial is not None:
                translated_symbols.append(fiducial.value)
                kept_indices.append(idx)

        if not kept_indices:
            return ECGAnnotation(
                symbol=np.array([], dtype=str),
                sample=np.array([], dtype=annotation.sample.dtype),
            )

        return ECGAnnotation(
            symbol=np.array(translated_symbols, dtype=str),
            sample=annotation.sample[np.array(kept_indices, dtype=np.int64)],
        )

    def __call__(self, annotation: ECGAnnotation) -> ECGAnnotation:
        return self.translate(annotation)

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> "FiducialMapper":
        symbol_to_fiducial: dict[str, Fiducials | GenericMarker] = {}

        for group_name, symbols in mapping.items():
            if group_name not in GROUP_TO_FIDUCIAL:
                raise KeyError(
                    f"Unknown annotation group {group_name!r}. "
                    f"Expected one of: {sorted(GROUP_TO_FIDUCIAL)}"
                )

            fiducial = GROUP_TO_FIDUCIAL[group_name]

            for symbol in symbols:
                symbol = str(symbol)
                existing = symbol_to_fiducial.get(symbol)

                if existing is not None and existing != fiducial:
                    raise ValueError(
                        f"Symbol {symbol!r} is mapped to both "
                        f"{existing.value!r} (group inferred earlier) and "
                        f"{fiducial.value!r} (group {group_name!r})."
                    )

                symbol_to_fiducial[symbol] = fiducial

        return cls(symbol_to_fiducial=symbol_to_fiducial)

    @classmethod
    def from_yaml(cls, filename: str = "annotation_labels.yaml") -> "FiducialMapper":
        return cls.from_dict(load_yaml(filename))
