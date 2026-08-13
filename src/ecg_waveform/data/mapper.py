from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ecg_waveform.config import load_yaml
from ecg_waveform.core import ECGAnnotation, Fiducials

GROUP_TO_FIDUCIAL: dict[str, Fiducials] = {
    "p_peaks": Fiducials.P_WAVE,
    "q_peaks": Fiducials.Q_WAVE,
    "r_peaks": Fiducials.R_WAVE,
    "s_peaks": Fiducials.S_WAVE,
    "t_peaks": Fiducials.T_WAVE,
}


@dataclass(frozen=True, slots=True)
class FiducialMapper:
    symbol_to_fiducial: dict[str, Fiducials] = field(default_factory=dict)

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

    def translate(self, annotation: ECGAnnotation) -> ECGAnnotation:
        if len(annotation) == 0:
            return annotation

        translated_symbols: list[str] = []
        kept_indices: list[int] = []

        for idx, raw_symbol in enumerate(annotation.symbol):
            fiducial = self.symbol_to_fiducial.get(str(raw_symbol))
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
        symbol_to_fiducial: dict[str, Fiducials] = {}

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
