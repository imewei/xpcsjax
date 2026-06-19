"""Regression tests: the L2 hierarchical Stage-1 warm-start logs must not be
misread as the fit's per-angle scaling mode.

RCA (2026-06-15): for ``two_component`` with ``per_angle_mode="auto"`` at
``n_phi >= constant_scaling_threshold`` the *final* fit resolves to ``averaged``
(2 scaling DOF). But ``_build_joint_problem`` runs an L2 Stage-1 physics-only
warm-start that delegates to ``_fit_joint_constant_multi_phi`` — a routine that
is *inherently per-angle* and logs ``Frozen per-angle scaling: contrast=[...]``
with one value per angle. Emitted at INFO with no warm-start context, those
lines read as if ``individual`` mode ran, contradicting the configured
``auto -> averaged``. The fit itself is correct (averaged); only the logs are
misleading. These tests pin the disambiguation: the warm-start must be labelled
as a warm-start naming the resolved final mode, and the per-angle arrays must
not leak to INFO during a warm-start. The genuine ``constant``-mode top-level
fit keeps its INFO banner (no-regression guard).
"""

from __future__ import annotations

import logging

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
    _fit_joint_constant_multi_phi,
)
from xpcsjax.optimization.nlsq.heterodyne_core import (
    _build_joint_problem,
    _fit_joint_averaged_multi_phi,
)

_CORE_LOGGER = "xpcsjax.optimization.nlsq.heterodyne_core"
_CONST_LOGGER = "xpcsjax.optimization.nlsq.heterodyne_constant_mode"


def _build_joint_with_logs(caplog, per_angle_mode, n_phi):
    model, c2, phi = make_synthetic_two_component(n_phi=n_phi, n_t=16)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": per_angle_mode,
            "enable_hierarchical": True,
        }
    )
    with (
        caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER),
        caplog.at_level(logging.DEBUG, logger=_CONST_LOGGER),
    ):
        _build_joint_problem(model, c2, phi, cfg, weights=None)
    return caplog.records


def test_l2_stage1_warmstart_banner_names_final_mode(caplog):
    # n_phi=3 with the default threshold (3) => auto resolves to averaged.
    records = _build_joint_with_logs(caplog, "auto", n_phi=3)
    # A SINGLE introductory banner must carry both the "warm-start" label and
    # the resolved final mode, so the per-angle quantile arrays that follow are
    # unambiguously the warm-start's, not the fit's scaling mode. (A stray
    # "warm-starting stage 2" line emitted AFTER the arrays does not count.)
    banner = [
        r.getMessage()
        for r in records
        if r.levelno >= logging.INFO
        and "warm-start" in r.getMessage().lower()
        and "averaged" in r.getMessage().lower()
    ]
    assert banner, (
        "Expected one INFO banner naming both 'warm-start' and the resolved "
        "final mode 'averaged'. Saw INFO lines:\n"
        + "\n".join(r.getMessage() for r in records if r.levelno >= logging.INFO)
    )


def test_l2_stage1_per_angle_arrays_not_at_info(caplog):
    records = _build_joint_with_logs(caplog, "auto", n_phi=3)
    offending = [
        r.getMessage()
        for r in records
        if r.levelno >= logging.INFO and "Frozen per-angle scaling" in r.getMessage()
    ]
    assert not offending, (
        "The per-angle 'Frozen per-angle scaling: contrast=[...]' line must be "
        "demoted to DEBUG during a Stage-1 warm-start (it reads as individual "
        f"mode at INFO). Leaked to INFO: {offending}"
    )


def test_averaged_path_stage1_warmstart_disambiguated(caplog):
    """The averaged-mode dispatch (`_fit_joint_averaged_multi_phi`) runs its own
    L2 Stage-1 warm-start via the same constant helper — it must disambiguate
    identically (banner names averaged; per-angle arrays not at INFO)."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_hierarchical": True,
        }
    )
    with (
        caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER),
        caplog.at_level(logging.DEBUG, logger=_CONST_LOGGER),
    ):
        _fit_joint_averaged_multi_phi(model, c2, phi, cfg, weights=None)
    info_records = [r for r in caplog.records if r.levelno >= logging.INFO]
    banner = [
        r.getMessage()
        for r in info_records
        if "warm-start" in r.getMessage().lower() and "averaged" in r.getMessage().lower()
    ]
    leaked = [r.getMessage() for r in info_records if "Frozen per-angle scaling" in r.getMessage()]
    assert banner, "averaged path missing the disambiguating warm-start banner"
    assert not leaked, f"averaged path leaked per-angle arrays to INFO: {leaked}"


def test_genuine_constant_fit_keeps_info_banner(caplog):
    """No-regression: a real top-level constant fit still logs at INFO."""
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=16)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "constant"})
    with caplog.at_level(logging.INFO, logger=_CONST_LOGGER):
        _fit_joint_constant_multi_phi(model, c2, phi, cfg, weights=None)
    info_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)
    assert "Frozen per-angle scaling" in info_text, (
        "The genuine constant-mode fit must keep its INFO per-angle banner; "
        "only the L2 warm-start caller should demote it."
    )
