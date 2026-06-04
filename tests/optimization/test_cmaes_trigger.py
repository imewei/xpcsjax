"""CMA-ES auto-triggers at scale_ratio >= 1000 (homodyne default).

XPCS multi-scale problems span >3 orders of magnitude (e.g., D0 ~ 1e4 vs
gamma_dot ~ 1e-3 → ratio ~ 1e7). This is the documented escape hatch; we
verify it directly so a regression localizes to the trigger function rather
than only surfacing via characterization."""

import inspect

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper


@pytest.fixture()
def wrapper():
    return CMAESWrapper()


def test_high_scale_ratio_triggers_cmaes(wrapper):
    """Realistic XPCS multi-scale bounds (D0 ~ 1e4 vs gamma_dot ~ 1e-3) must trigger.

    scale_ratio here measures spread of parameter widths across the parameter
    vector — not range within a single parameter. Three params spanning 9
    orders of magnitude reliably clear the 1000 threshold."""
    lower = np.array([1.0e2, 1.0e-4, -0.5])
    upper = np.array([5.0e4, 1.0, 0.5])
    assert wrapper.should_use_cmaes((lower, upper)), (
        f"multi-scale XPCS bounds must enable CMA-ES "
        f"(scale_ratio={wrapper.compute_scale_ratio((lower, upper))})"
    )


def test_low_scale_ratio_does_not_trigger(wrapper):
    """Tightly-clustered parameter widths must NOT enable CMA-ES.

    All three parameters with width ≈ 1 — scale_ratio = 1, below 1000 threshold."""
    lower = np.array([1.0, 2.0, 3.0])
    upper = np.array([2.0, 3.0, 4.0])
    assert not wrapper.should_use_cmaes((lower, upper)), (
        f"unimodal-scale bounds must not enable CMA-ES "
        f"(scale_ratio={wrapper.compute_scale_ratio((lower, upper))})"
    )


def test_default_threshold_is_1000():
    """The documented default scale_threshold is 1000.0."""
    sig = inspect.signature(CMAESWrapper.should_use_cmaes)
    threshold_param = next(
        (
            p
            for name, p in sig.parameters.items()
            if "threshold" in name.lower() or "scale_thr" in name.lower()
        ),
        None,
    )
    assert threshold_param is not None, (
        "CMAESWrapper.should_use_cmaes has no threshold parameter — "
        "homodyne's documented API has scale_threshold=1000.0 by default."
    )
    assert threshold_param.default == pytest.approx(1000.0), (
        f"default scale_threshold drifted from documented 1000.0 to {threshold_param.default}"
    )


def test_compute_scale_ratio_increases_with_spread(wrapper):
    """compute_scale_ratio reports parameter-width spread; wider spread → higher ratio.

    Two parameters of width 1 → ratio = 1. Add a parameter of width 1e6 →
    ratio explodes. Verifies the spread metric responds monotonically."""
    tight_lower = np.array([1.0, 2.0])
    tight_upper = np.array([2.0, 3.0])
    tight_ratio = wrapper.compute_scale_ratio((tight_lower, tight_upper))

    wide_lower = np.array([1.0, 1.0e-4])
    wide_upper = np.array([2.0, 1.0e3])
    wide_ratio = wrapper.compute_scale_ratio((wide_lower, wide_upper))

    assert wide_ratio > tight_ratio * 100, (
        f"compute_scale_ratio should respond to width spread: "
        f"tight={tight_ratio}, wide={wide_ratio}"
    )
