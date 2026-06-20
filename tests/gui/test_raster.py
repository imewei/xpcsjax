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


def test_thin_array_does_not_collapse_to_zero_dim():
    # A 2-row array longer than max_dim: step = ceil(2050/1024) = 3, so the short
    # axis trims to (2 // 3) * 3 == 0 rows. The reshape then yields a zero-height
    # array, which breaks ImageItem.setImage. The function must keep both axes > 0.
    a = np.random.default_rng(0).random((2, 2050))
    out = rasterize(a, max_dim=1024)
    assert out.ndim == 2
    assert out.shape[0] > 0 and out.shape[1] > 0


def test_infinities_are_sanitized_for_display():
    # ±inf in the source poisons block means (inf) and ImageItem colormap
    # autoscaling. rasterize must hand back an all-finite numeric array.
    a = np.ones((2000, 2000))
    a[0, 0] = np.inf
    a[1, 1] = -np.inf
    out = rasterize(a, max_dim=512)
    assert np.isfinite(out).all()


def test_nan_is_preserved_for_masked_regions():
    # NaN denotes masked/excluded pixels (ImageItem renders it transparent); the
    # finite-sanitization must NOT overwrite it (only ±inf is clipped).
    a = np.ones((2000, 2000))
    a[0, 0] = np.nan
    out = rasterize(a, max_dim=512)
    assert np.isnan(out).any()
