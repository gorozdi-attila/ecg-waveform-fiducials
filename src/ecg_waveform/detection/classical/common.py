import numpy as np


def first_below_persistent(
    values: np.ndarray,
    start: int,
    stop: int,
    step: int,
    threshold: float,
    persistence: int,
) -> int | None:
    if step == 0:
        raise ValueError("step must not be zero")

    persistence = max(1, int(persistence))
    start, stop, step = int(start), int(stop), int(step)
    n = len(values)

    if n == 0:
        return None

    if (step > 0 and start > stop) or (step < 0 and start < stop):
        return None

    last_offset = step * (persistence - 1)

    idx = start
    while idx <= stop if step > 0 else idx >= stop:
        last_idx = idx + last_offset

        if step > 0:
            if last_idx > stop:
                break
        else:
            if last_idx < stop:
                break

        if idx < 0 or idx >= n or last_idx < 0 or last_idx >= n:
            break

        indices = idx + step * np.arange(persistence)
        if np.all(values[indices] < threshold):
            return idx

        idx += step

    return None


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values)
    width = max(1, int(width))

    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")

    if len(values) == 0 or width == 1:
        return values.astype(np.result_type(values.dtype, np.float64), copy=True)

    left = (width - 1) // 2
    right = width // 2
    padded = np.pad(values, (left, right), mode="edge")

    kernel = np.full(width, 1.0 / width)
    return np.convolve(padded, kernel, mode="valid")
