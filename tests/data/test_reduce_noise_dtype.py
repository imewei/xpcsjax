"""Regression test for integer-dtype truncation in noise reduction.

Bug: ``_reduce_noise`` builds its working buffer with ``np.array(v)``
(preserving the source dtype) and then assigns *float* filter results
(gaussian/wiener/savgol) back into ``denoised_data["c2_exp"][i]``. If the
source ``c2_exp`` is an integer dtype, that item-assignment silently
truncates the fractional filtered output -- the same bug class already fixed
in ``_correct_diagonal_enhanced`` / ``_normalize_data``.
"""

import numpy as np
import pytest

from xpcsjax.data.preprocessing import PreprocessingPipeline


def _integer_c2():
    n = 8
    rng = np.random.default_rng(0)
    base = rng.integers(1, 5, size=(n, n)).astype(np.int64)
    return (base + base.T) // 2  # symmetric-ish, dtype stays int64


@pytest.mark.parametrize("method", ["gaussian", "wiener", "savgol"])
def test_reduce_noise_upcasts_integer_c2_no_truncation(method):
    pytest.importorskip("scipy")
    pipeline = PreprocessingPipeline({})
    matrix = _integer_c2()
    data = {"c2_exp": np.stack([matrix])}  # shape (1, 8, 8), int dtype

    config: dict = {"method": method}
    if method == "savgol":
        config.update(window_length=3, polyorder=1)

    denoised = pipeline._reduce_noise(data, config)

    c2 = np.asarray(denoised["c2_exp"])
    assert np.issubdtype(c2.dtype, np.floating), (
        f"c2_exp not upcast to float for method={method}: dtype={c2.dtype}"
    )
    frac = np.abs(c2 - np.round(c2))
    assert np.max(frac) > 1e-9, f"{method} output was truncated to integers"
