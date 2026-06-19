"""Tests for numeric block-mean display rasterization."""

import numpy as np

from xpcsjax.gui.views.raster import rasterize


def test_small_array_is_returned_unchanged():
    a = np.ones((10, 10))
    out = rasterize(a, max_dim=1024)
    assert out.shape == (10, 10)


def test_large_array_is_downsampled_within_max_dim():
    a = np.random.default_rng(0).random((4000, 4000))
    out = rasterize(a, max_dim=1000)
    assert max(out.shape) <= 1000
    assert a.shape == (4000, 4000)  # input untouched
