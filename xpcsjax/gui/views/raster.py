"""Display-only block-mean rasterization of large 2-D arrays (numeric).

Rendering decimation for the GUI — NOT analysis downsampling (the fit always
uses full-resolution data on disk).
"""

from __future__ import annotations

import numpy as np


def rasterize(array: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Block-mean downsample a 2-D array to <= ``max_dim`` per side (numeric).

    Returns a **numeric** array (not a colored image), so PyQtGraph's
    ``ImageItem`` keeps applying its own interactive colormap/levels. This is
    display rasterization only — never analysis downsampling; full-resolution
    data stays on disk. (Datashader's colored-image fast path lives in
    ``xpcsjax.viz`` for the *static publication export*; it returns an RGB image,
    unsuitable for the interactive numeric ImageItem, so it is intentionally not
    used here.) Small arrays pass through untouched.
    """
    if max_dim <= 0:
        raise ValueError(f"max_dim must be positive, got {max_dim}")
    array = np.asarray(array, dtype=float)
    # Sanitize ±inf BEFORE any reduction: an infinity poisons the block mean and
    # breaks ImageItem's colormap autoscaling. Clip each infinity to the finite
    # data range; leave NaN intact (it marks masked pixels — ImageItem renders it
    # transparent). All-non-finite input falls back to a flat zero array.
    if not np.isfinite(array).all():
        finite = array[np.isfinite(array)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
        array = np.where(np.isposinf(array), hi, array)
        array = np.where(np.isneginf(array), lo, array)
    if array.ndim != 2 or max(array.shape) <= max_dim:
        return array
    step = int(np.ceil(max(array.shape) / max_dim))
    # Trim to a multiple of `step`, then mean over step x step blocks.
    h = (array.shape[0] // step) * step
    w = (array.shape[1] // step) * step
    if h == 0 or w == 0:
        # A very thin array (one axis < step) cannot be block-decimated without
        # collapsing to a zero-length axis — pass it through untouched (it is
        # already small on that axis and needs no decimation there).
        return array
    trimmed = array[:h, :w]
    return trimmed.reshape(h // step, step, w // step, step).mean(axis=(1, 3))
