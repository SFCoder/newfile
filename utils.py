import numpy as np


def summarize_numbers(numbers):
    """Return min, max, mean, and median of a list of numbers."""
    arr = np.array(numbers, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }
