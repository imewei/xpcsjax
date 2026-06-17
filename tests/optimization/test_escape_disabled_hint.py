"""Actionable hint when a heterodyne joint fit fails with NO global escape enabled.

C044 ``two_component`` RCA (2026-06-17): a degenerate 14-D joint fit failed to
converge and finalized a poor result, but the run had ``cmaes.enable=false`` (and
no multistart), so the CMA-ES global escape that would rescue it never ran —
``Per-angle dispatch: ... escape=None``. Nothing in the failure output pointed the
user at the fix. This hint closes that gap: on a FAILED joint fit with no escape
enabled, log an actionable "enable cmaes.enable + n_seeds>=3" message. Strictly
diagnostic — no numeric effect.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from xpcsjax.optimization.nlsq import heterodyne_core as hc

_LOGGER = "xpcsjax.optimization.nlsq.heterodyne_core"


def test_should_hint_when_failed_and_no_escape():
    assert (
        hc._should_hint_enable_escape(success=False, enable_cmaes=False, multistart=False)
        is True
    )


def test_no_hint_when_fit_succeeded():
    assert (
        hc._should_hint_enable_escape(success=True, enable_cmaes=False, multistart=False)
        is False
    )


def test_no_hint_when_cmaes_already_enabled():
    assert (
        hc._should_hint_enable_escape(success=False, enable_cmaes=True, multistart=False)
        is False
    )


def test_no_hint_when_multistart_already_enabled():
    assert (
        hc._should_hint_enable_escape(success=False, enable_cmaes=False, multistart=True)
        is False
    )


def test_log_hint_emits_actionable_cmaes_message(caplog):
    """A failed escape-less fit logs a hint naming cmaes.enable and n_seeds."""
    result = SimpleNamespace(success=False)
    config = SimpleNamespace(enable_cmaes=False, multistart=False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        hc.log_enable_escape_hint(result, config)
    text = " ".join(r.getMessage() for r in caplog.records).lower()
    assert "cmaes" in text and "n_seeds" in text, (
        "hint must name the cmaes escape and the n_seeds knob"
    )


def test_log_hint_silent_when_fit_succeeded(caplog):
    result = SimpleNamespace(success=True)
    config = SimpleNamespace(enable_cmaes=False, multistart=False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        hc.log_enable_escape_hint(result, config)
    assert not caplog.records, "a converged fit must not emit the enable-escape hint"


def test_log_hint_silent_when_escape_already_enabled(caplog):
    result = SimpleNamespace(success=False)
    config = SimpleNamespace(enable_cmaes=True, multistart=False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        hc.log_enable_escape_hint(result, config)
    assert not caplog.records, "no hint when the escape is already enabled"


def test_log_hint_robust_to_missing_attributes(caplog):
    """Missing ``success`` defaults to converged (no hint); the helper never raises."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        hc.log_enable_escape_hint(SimpleNamespace(), SimpleNamespace())
    assert not caplog.records
