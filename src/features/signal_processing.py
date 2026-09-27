import numpy as np
from scipy import signal


def butterworth_filter(
    x: np.ndarray,
    lowcut: float | None = 0.5,
    highcut: float | None = 45.0,
    fs: float = 125.0,
    order: int = 3,
) -> np.ndarray:
    """Apply a Butterworth filter along the time axis."""
    nyquist = 0.5 * fs
    if lowcut is not None and highcut is not None:
        btype = "bandpass"
        wn = [lowcut / nyquist, highcut / nyquist]
    elif lowcut is not None:
        btype = "highpass"
        wn = lowcut / nyquist
    elif highcut is not None:
        btype = "lowpass"
        wn = highcut / nyquist
    else:
        return x.astype(np.float32, copy=True)

    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, x, axis=1).astype(np.float32)


def dominant_peak_features(x: np.ndarray) -> np.ndarray:
    """Return main peak location, height, prominence, and width."""
    features = np.zeros((x.shape[0], 4), dtype=np.float32)

    for i, row in enumerate(x):
        peaks, properties = signal.find_peaks(row, prominence=0.02)
        if len(peaks) == 0:
            peak_idx = int(np.argmax(row))
            features[i] = [
                peak_idx / max(len(row) - 1, 1),
                row[peak_idx],
                0.0,
                0.0,
            ]
            continue

        prominences = properties["prominences"]
        best = int(np.argmax(prominences))
        peak_idx = int(peaks[best])
        widths = signal.peak_widths(row, [peak_idx], rel_height=0.5)[0]
        features[i] = [
            peak_idx / max(len(row) - 1, 1),
            row[peak_idx],
            prominences[best],
            widths[0] / max(len(row) - 1, 1),
        ]

    return features
