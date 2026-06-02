"""Data-integrity guards on the joint global-escape keep-better decision.

These pin the fix for the quality-gate finding that a non-finite warm-start SSR
silently beat a *finite* escape candidate: ``cand_ssr <= nan * (1 + eps)`` is
always ``False``, so a NaN/Inf warm-start fit was "kept" over a real escape
result and returned tagged as a success. The keep-better decision is now a pure,
NaN-aware helper so the rule lives in one place and is unit-testable without
running CMA-ES.
"""

from __future__ import annotations

from xpcsjax.optimization.nlsq.heterodyne_core import _escape_keeps_candidate


def test_finite_candidate_beats_nonfinite_warm_start():
    # The bug: a NaN warm SSR must NEVER win over a finite candidate.
    assert _escape_keeps_candidate(ssr_warm=float("nan"), ssr_cand=1.0) is True
    assert _escape_keeps_candidate(ssr_warm=float("inf"), ssr_cand=1.0) is True


def test_nonfinite_candidate_never_kept():
    # A NaN/Inf candidate is never an improvement, even over a NaN warm start.
    assert _escape_keeps_candidate(ssr_warm=1.0, ssr_cand=float("nan")) is False
    assert _escape_keeps_candidate(ssr_warm=float("nan"), ssr_cand=float("inf")) is False
    assert _escape_keeps_candidate(ssr_warm=float("nan"), ssr_cand=float("nan")) is False


def test_finite_keep_better_semantics_unchanged():
    # Strictly-better candidate kept; worse candidate rejected; ties kept
    # (within the 1e-12 tolerance the original comparison used).
    assert _escape_keeps_candidate(ssr_warm=2.0, ssr_cand=1.0) is True
    assert _escape_keeps_candidate(ssr_warm=1.0, ssr_cand=2.0) is False
    assert _escape_keeps_candidate(ssr_warm=1.0, ssr_cand=1.0) is True
