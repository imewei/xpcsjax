"""Regression tests for xpcsjax.core.heterodyne_scaling_utils.

estimate_contrast_offset_from_quantiles separates the C2 floor (large-lag) from
the ceiling (small-lag) using percentiles of ``delta_t``. A non-finite delta_t
value would make ``np.percentile(dt, ...)`` return NaN, collapsing both lag
masks to all-False and silently degrading the estimate to undifferentiated
global quantiles. The finite filter must mask BOTH arrays so a NaN delta_t point
is dropped rather than poisoning the lag thresholds.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.core.heterodyne_scaling_utils import (
    estimate_contrast_offset_from_quantiles,
)


def _lag_separated_dataset():
    """C2 that genuinely decays with lag, so lag-separation differs from a global
    quantile collapse. Returns (c2, dt) with >= 100 finite points."""
    n = 200
    dt = np.linspace(0.0, 10.0, n)
    # offset 1.0, contrast 0.5: high near dt=0, decaying to ~offset at large dt
    c2 = 1.0 + 0.5 * np.exp(-dt)
    return c2, dt


def test_nan_delta_t_is_dropped_not_poisoning_thresholds():
    c2, dt = _lag_separated_dataset()
    baseline = estimate_contrast_offset_from_quantiles(c2, dt)

    # Inject one finite-c2 / NaN-dt pair. With paired finite filtering this point
    # is dropped and the estimate is unchanged (one point out of 200 is
    # negligible to the percentiles). Without the dt mask, percentile(dt) → NaN,
    # both lag masks go all-False, and the estimate collapses to global quantiles.
    c2_bad = np.append(c2, 1.25)
    dt_bad = np.append(dt, np.nan)
    got = estimate_contrast_offset_from_quantiles(c2_bad, dt_bad)

    np.testing.assert_allclose(got, baseline, rtol=1e-6, atol=1e-6)


def test_all_finite_inputs_unchanged():
    """The added dt mask must be a no-op when every value is finite."""
    c2, dt = _lag_separated_dataset()
    # Idempotent: calling twice yields the same result; no exception, finite out.
    contrast, offset = estimate_contrast_offset_from_quantiles(c2, dt)
    assert np.isfinite(contrast) and np.isfinite(offset)
    assert 0.0 <= contrast <= 1.0
    assert 0.5 <= offset <= 1.5
