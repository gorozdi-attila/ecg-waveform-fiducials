from typing import Protocol

import numpy as np

from ecg_waveform.core import ECGAnnotation, ECGSignal


class BaseDelineator(Protocol):
    def delineate(
        self,
        signal: ECGSignal,
        r_samples: np.ndarray,
        qrs_band: np.ndarray | None = None,
    ) -> ECGAnnotation: ...
