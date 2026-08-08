from dataclasses import dataclass

import numpy as np

from ecg_waveform.core import ECGAnnotation, ECGSignal, Fiducials
from ecg_waveform.signal.preprocessing import butterworth_filter


@dataclass(slots=True)
class WindowedFiducialDelineator:
    qs_search_window_s: float = 0.05

    wide_band_low_cutoff: float = 0.5
    wide_band_high_cutoff: float = 40.0
    wide_band_order: int = 4

    p_wave_window_s: tuple[float, float] = (0.24, 0.08)

    t_wave_window_s: tuple[float, float] = (0.10, 0.40)

    def _wide_band(self, signal: ECGSignal) -> np.ndarray:
        return butterworth_filter(
            signal=signal,
            low_cutoff=self.wide_band_low_cutoff,
            high_cutoff=self.wide_band_high_cutoff,
            order=self.wide_band_order,
        ).sample

    def _detect_q_samples(
        self, qrs_band: np.ndarray, sample_rate: int, r_samples: np.ndarray
    ) -> np.ndarray:
        window = max(1, int(self.qs_search_window_s * sample_rate))
        q_samples = []

        for r in r_samples:
            start = max(0, r - window)
            segment = qrs_band[start:r]

            if len(segment) == 0:
                continue

            q_samples.append(start + int(np.argmin(segment)))

        return np.array(q_samples, dtype=np.int64)

    def _detect_s_samples(
        self, qrs_band: np.ndarray, sample_rate: int, r_samples: np.ndarray
    ) -> np.ndarray:
        window = max(1, int(self.qs_search_window_s * sample_rate))
        s_samples = []

        for r in r_samples:
            end = min(len(qrs_band), r + 1 + window)
            segment = qrs_band[r + 1 : end]

            if len(segment) == 0:
                continue

            s_samples.append(r + 1 + int(np.argmin(segment)))

        return np.array(s_samples, dtype=np.int64)

    def _detect_p_samples(
        self, wide_band: np.ndarray, sample_rate: int, r_samples: np.ndarray
    ) -> np.ndarray:
        max_offset = int(self.p_wave_window_s[0] * sample_rate)
        min_offset = int(self.p_wave_window_s[1] * sample_rate)

        p_samples = []

        for r in r_samples:
            start = max(0, r - max_offset)
            end = max(0, r - min_offset)

            if end <= start:
                continue

            segment = wide_band[start:end]
            if len(segment) == 0:
                continue

            p_samples.append(start + int(np.argmax(segment)))

        return np.array(p_samples, dtype=np.int64)

    def _detect_t_samples(
        self, wide_band: np.ndarray, sample_rate: int, r_samples: np.ndarray
    ) -> np.ndarray:
        min_offset = int(self.t_wave_window_s[0] * sample_rate)
        max_offset = int(self.t_wave_window_s[1] * sample_rate)

        t_samples = []

        for r in r_samples:
            start = min(len(wide_band), r + min_offset)
            end = min(len(wide_band), r + max_offset)

            if end <= start:
                continue

            segment = wide_band[start:end]
            if len(segment) == 0:
                continue

            t_samples.append(start + int(np.argmax(segment)))

        return np.array(t_samples, dtype=np.int64)

    def delineate(
        self,
        signal: ECGSignal,
        r_samples: np.ndarray,
        qrs_band: np.ndarray | None = None,
    ) -> ECGAnnotation:
        qs_source = qrs_band if qrs_band is not None else signal.sample
        wide_band = self._wide_band(signal)

        q_samples = self._detect_q_samples(qs_source, signal.sample_rate, r_samples)
        s_samples = self._detect_s_samples(qs_source, signal.sample_rate, r_samples)
        p_samples = self._detect_p_samples(wide_band, signal.sample_rate, r_samples)
        t_samples = self._detect_t_samples(wide_band, signal.sample_rate, r_samples)

        samples = np.concatenate([p_samples, q_samples, s_samples, t_samples])
        symbols = np.array(
            [Fiducials.P_WAVE.value] * len(p_samples)
            + [Fiducials.Q_WAVE.value] * len(q_samples)
            + [Fiducials.S_WAVE.value] * len(s_samples)
            + [Fiducials.T_WAVE.value] * len(t_samples),
            dtype=str,
        )

        order = np.argsort(samples, kind="stable")

        return ECGAnnotation(symbol=symbols[order], sample=samples[order])
