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
    array = np.asarray(array)
    if array.ndim != 2 or max(array.shape) <= max_dim:
        return array
    step = int(np.ceil(max(array.shape) / max_dim))
    # Trim to a multiple of `step`, then mean over step x step blocks.
    h = (array.shape[0] // step) * step
    w = (array.shape[1] // step) * step
    trimmed = array[:h, :w]
    return trimmed.reshape(h // step, step, w // step, step).mean(axis=(1, 3))
