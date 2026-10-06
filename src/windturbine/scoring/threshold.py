"""Anomaly score and threshold. Copied from the arm notebooks; constants became arguments."""
from __future__ import annotations

import numpy as np


def reconstruction_error(model, X, batch_size: int = 8192) -> np.ndarray:
    """L2 norm of the residual per row. Calls the model directly rather than
    .predict(): in a per-event loop, .predict() retraces its tf.function on every
    new array shape and floods the log. ``model`` is any callable returning an
    object with ``.numpy()`` (a Keras model)."""
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch_size):
        chunk = X[i : i + batch_size]
        rec = model(chunk, training=False).numpy()
        out[i : i + len(chunk)] = np.linalg.norm(chunk - rec, axis=1)
    return out


def pick_threshold(val_scores, q: float = 0.99, tag: str = ""):
    """Quantile threshold plus a guard: a validation block containing a regime the
    model never saw produces a few enormous residuals that drag the quantile up and
    silently disable detection for that event."""
    thr = float(np.quantile(val_scores, q))
    med = float(np.median(val_scores))
    ratio = thr / max(med, 1e-9)
    if ratio > 10:
        print(
            f"    WARNING{' ' + tag if tag else ''}: threshold is {ratio:.0f}x the "
            f"median validation score (med={med:.2f}, thr={thr:.2f}). The "
            f"validation block likely contains an unseen operating regime."
        )
    return thr, {
        "thr": thr,
        "val_med": med,
        "val_p99": float(np.quantile(val_scores, 0.99)),
        "val_max": float(val_scores.max()),
        "thr_over_median": ratio,
    }
