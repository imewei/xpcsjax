"""Phase-0 unit tests for the single per-angle-mode resolver seam (spec §4 Seam 1)."""

from __future__ import annotations

import pytest

from xpcsjax.optimization.nlsq.per_angle_mode import (
    DEFAULT_CONSTANT_SCALING_THRESHOLD,
    n_optimized,
    resolve_per_angle_mode,
)


def test_default_constant_scaling_threshold_is_three():
    assert DEFAULT_CONSTANT_SCALING_THRESHOLD == 3


@pytest.mark.parametrize("token", ["constant", "averaged", "individual"])
def test_explicit_modes_pass_through_identity(token):
    # Resolved modes resolve to themselves regardless of n_phi / threshold.
    for n_phi in (1, 2, 3, 10, 23):
        assert resolve_per_angle_mode(token, n_phi) == token


@pytest.mark.parametrize(
    ("n_phi", "expected"),
    [
        (1, "individual"),
        (2, "individual"),
        (3, "averaged"),
        (4, "averaged"),
        (23, "averaged"),
    ],
)
def test_auto_resolves_by_default_threshold(n_phi, expected):
    # auto -> averaged when n_phi >= DEFAULT_CONSTANT_SCALING_THRESHOLD (3) else individual.
    assert resolve_per_angle_mode("auto", n_phi) == expected


def test_auto_honors_custom_threshold():
    assert resolve_per_angle_mode("auto", 5, constant_scaling_threshold=6) == "individual"
    assert resolve_per_angle_mode("auto", 6, constant_scaling_threshold=6) == "averaged"


@pytest.mark.parametrize("bad", ["Auto", "", "AVERAGED", "foo"])
def test_unknown_and_removed_tokens_raise_valueerror(bad):
    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        resolve_per_angle_mode(bad, n_phi=10)


@pytest.mark.parametrize(
    ("mode", "n_phi", "expected"),
    [
        ("constant", 23, 0),
        ("averaged", 23, 2),
        ("averaged", 1, 2),
        ("individual", 1, 2),
        ("individual", 23, 46),
    ],
)
def test_n_optimized_truth_table(mode, n_phi, expected):
    assert n_optimized(mode, n_phi) == expected


def test_n_optimized_rejects_unresolved_token():
    with pytest.raises(ValueError, match="unknown per_angle_mode"):
        n_optimized("auto", 10)  # type: ignore[arg-type]
