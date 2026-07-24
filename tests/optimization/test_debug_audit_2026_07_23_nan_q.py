"""Regression: scalar q-extraction sites must reject NaN, not silently poison
the JAX physics model.

Finding #1 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md:
xpcs_loader.py's _validate_loaded_arrays now tolerates NaN in
wavevector_q_list (bad-pixel masking is legitimate). But three call sites
extract wavevector_q_list[0] as a bare scalar q and feed it straight into the
JAX physics model with no guard. A bad-pixel NaN landing at index 0 (or being
the only value reaching these extraction sites) must raise, not silently
produce a NaN-poisoned fit.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.adapter import NLSQAdapter
from xpcsjax.optimization.nlsq.core import _normalize_data_to_object


def test_normalize_data_to_object_rejects_nan_q():
    data = {
        "phi_angles_list": np.array([0.0, 45.0]),
        "c2_exp": np.ones((2, 4, 4)),
        "wavevector_q_list": np.array([np.nan, 0.02]),
    }
    with pytest.raises(ValueError, match="wavevector_q_list"):
        _normalize_data_to_object(data, config=object(), logger=logging.getLogger("t"))


def test_normalize_data_to_object_accepts_finite_q():
    data = {
        "phi_angles_list": np.array([0.0, 45.0]),
        "c2_exp": np.ones((2, 4, 4)),
        "wavevector_q_list": np.array([0.02, np.nan]),  # NaN elsewhere is fine
    }
    obj = _normalize_data_to_object(data, config=object(), logger=logging.getLogger("t"))
    assert obj.q == pytest.approx(0.02)


def test_build_model_function_rejects_nan_q(monkeypatch):
    adapter = NLSQAdapter.__new__(NLSQAdapter)  # bypass __init__, only need _build_model_function
    data = {
        "wavevector_q_list": np.array([np.nan]),
        "phi_angles_list": np.array([0.0, 45.0]),
    }
    with pytest.raises(ValueError, match="wavevector_q_list"):
        adapter._build_model_function(
            data,
            config=object(),
            analysis_mode=None,
            per_angle_scaling=False,
            n_phi=2,
        )
