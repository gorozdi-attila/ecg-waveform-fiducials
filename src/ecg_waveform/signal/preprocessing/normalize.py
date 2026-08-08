import numpy as np

from ecg_waveform.core import ECGSignal


def z_score_normalize(signal: ECGSignal) -> ECGSignal:
    mean = np.mean(signal.sample)
    std = np.std(signal.sample)

    if std == 0:
        normalized = signal.sample - mean
    else:
        normalized = (signal.sample - mean) / std

    return signal.with_sample(normalized)
