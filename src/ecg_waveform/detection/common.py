import numpy as np


def first_below_persistent(
    values: np.ndarray,
    start: int,
    stop: int,
    step: int,
    threshold: float,
    persistence: int,
) -> int | None:
    persistence = max(1, int(persistence))
    idx = int(start)

    while idx >= stop if step < 0 else idx <= stop:
        indices = idx + step * np.arange(persistence)

        if np.any(indices < 0) or np.any(indices >= len(values)):
            break

        if step < 0:
            if np.any(indices < stop):
                break
        else:
            if np.any(indices > stop):
                break

        if np.all(values[indices] < threshold):
            return idx

        idx += step

    return None


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))

    if width <= 1 or len(values) < width:
        return values.copy()

    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")
