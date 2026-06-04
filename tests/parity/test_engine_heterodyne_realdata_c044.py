"""Task #15 — REAL-DATA (C044) engine-vs-production fit-parity, maintainer-gated.

This is the real-noisy-data counterpart to
``tests/parity/test_engine_heterodyne_fit_parity.py`` (which proves the finding on
a NOISELESS well-posed fixture where the true minimum is known). Here we answer
the gating question on REAL C044 ``two_component`` data where the true minimum is
UNKNOWN:

    Does ``chi2_engine <= chi2_production`` (no-worse) hold, or was the
    noiseless engine advantage a fixture artifact?

THE HONEST FINDING (see scripts/realdata_engine_fit_parity_c044.py for the full
measured numbers across n_t ∈ {48,64,80} and angle subsets):

* ``fixed_constant`` — STRICT parity on real data (rel_diff ~1e-16, machine
  zero), exactly as on the noiseless fixture: with the SAME frozen scaling both
  sides solve the identical physics-only problem.

* ``individual`` — the DRAMATIC noiseless advantage (engine ~1e-15 vs production
  trapped at SSR ~7e-2) does NOT reproduce on real noisy data. The two paths are
  NEAR-TIED: |rel_diff| ≲ 8e-4, and the SIGN of the tiny gap FLIPS by subset
  (engine better at n_t=48/80, marginally worse at n_t=64 by +8e-4 — verified
  NOT budget-limited: identical at nfev=500 and nfev=1000). The engine is
  therefore NO-WORSE within ~1e-3 but NOT strictly better on real data.

This test asserts only the durable, honest contract: **engine no-worse within a
1e-3 relative tolerance** (the same keep-better tolerance the production global
escapes use), and STRICT parity for ``fixed_constant``. It does NOT assert the
noiseless "engine reaches ~0" clause — on real noisy data there is a finite
irreducible residual and no known minimum to reach.

GATING — maintainer-local LIVE data oracle, OFF by default
----------------------------------------------------------
Reads the real C044 dataset at ``${XPCSJAX_DATA_ROOT}/C044/xpcsjax_config.yaml``
(default data root ``/home/wei/Documents/Projects/data``). Skips unless the
config + its cached data are present, so a fresh clone / CI never runs it. To run
locally::

    XPCSJAX_DATA_ROOT=/home/wei/Documents/Projects/data \
        uv run pytest tests/parity/test_engine_heterodyne_realdata_c044.py -v

Touches NO production code — it only READS the production dispatch and the engine
modules, exactly like the noiseless sibling.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

# Resolve the C044 config under the maintainer data root. Presence of BOTH the
# config and its cached npz is required (the loader needs the cache to be fast).
_DATA_ROOT = os.environ.get("XPCSJAX_DATA_ROOT", "/home/wei/Documents/Projects/data")
_C044_DIR = Path(_DATA_ROOT) / "C044"
_C044_CONFIG = _C044_DIR / "xpcsjax_config.yaml"
_C044_CACHE = _C044_DIR / "cached_frames_1000_2000.npz"

_GATE_OK = _C044_CONFIG.is_file() and _C044_CACHE.is_file()
_SKIP_REASON = (
    "real-data C044 oracle is maintainer-local; set XPCSJAX_DATA_ROOT to a tree "
    f"containing C044/xpcsjax_config.yaml + its cache (looked in {_C044_DIR}). "
    "Never enabled in CI / fresh clones."
)

# Import the script helpers by path (the script lives under scripts/, not a
# package) so we do not duplicate the engine-route construction.
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "realdata_engine_fit_parity_c044.py"
)


def _load_helpers():
    spec = importlib.util.spec_from_file_location("_realdata_c044_helpers", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Small time window keeps the in-memory joint fit to a few seconds per mode while
# staying on REAL noisy data (verified stable across n_t ∈ {48,64,80}).
_N_T = 48
_NFEV = 400
# No-worse relative tolerance — the same 1e-3 keep-better band the production
# global escapes use. fixed_constant additionally asserted at strict 1e-6.
_NO_WORSE_RTOL = 1e-3


@pytest.fixture(scope="module")
def _realdata_results():
    mod = _load_helpers()
    model, c2, phi, info = mod.load_real_subset(
        str(_C044_CONFIG), n_t=_N_T, n_phi=0
    )
    out = {}
    for mode in ("fixed_constant", "individual"):
        out[mode] = mod.run_reference_and_engine(model, c2, phi, mode=mode, nfev=_NFEV)
    out["_info"] = info
    return out


@pytest.mark.skipif(not _GATE_OK, reason=_SKIP_REASON)
def test_realdata_subset_is_real_and_nonempty(_realdata_results):
    """Sanity: we are fitting a real, non-degenerate C044 subset (not a fixture)."""
    info = _realdata_results["_info"]
    assert info["n_phi_used"] >= 3, "need multiple angles for the individual mode"
    assert info["n_t_used"] == _N_T
    assert info["total_points_used"] > 10_000


@pytest.mark.skipif(not _GATE_OK, reason=_SKIP_REASON)
def test_realdata_fixed_constant_strict_parity(_realdata_results):
    """``fixed_constant`` — STRICT objective parity on REAL noisy C044 data.

    With identical frozen per-angle scaling both sides solve the same physics-only
    problem and reach the same minimum (machine-zero rel_diff). This mirrors the
    noiseless sibling's strict-parity result — it is NOT a noiseless artifact.
    """
    out = _realdata_results["fixed_constant"]
    chi2_ref, chi2_engine = out["chi2_ref"], out["chi2_engine"]
    assert np.isfinite(chi2_ref) and np.isfinite(chi2_engine)
    rel = abs(chi2_engine - chi2_ref) / max(abs(chi2_ref), 1e-300)
    assert np.isclose(chi2_engine, chi2_ref, rtol=1e-6, atol=0.0), (
        f"fixed_constant real-data: engine {chi2_engine!r} != production "
        f"{chi2_ref!r} (rel_diff={rel:.3e}). With identical frozen scaling the two "
        "physics-only solves must reach the same minimum; a mismatch is a real "
        "residual/scaling/layout/solver bug. Do NOT loosen this; diagnose it."
    )


@pytest.mark.skipif(not _GATE_OK, reason=_SKIP_REASON)
def test_realdata_individual_engine_no_worse(_realdata_results):
    """``individual`` — engine NO-WORSE than production on REAL noisy C044 data.

    The honest real-data finding: the engine route is no-worse within 1e-3
    relative, but (unlike the noiseless fixture) it does NOT strictly beat
    production — the tiny gap flips sign by subset. We assert ONLY no-worse; we do
    NOT assert the noiseless "reaches ~0" clause (there is no known minimum on
    real noisy data).

    A STRICTLY-worse engine beyond the 1e-3 keep-better band WOULD be a real
    residual/scaling/layout/solver regression. Do NOT loosen this band to mask
    such a regression.
    """
    out = _realdata_results["individual"]
    chi2_ref, chi2_engine = out["chi2_ref"], out["chi2_engine"]
    assert np.isfinite(chi2_ref) and np.isfinite(chi2_engine)
    assert out["ref_convergence_status"] == "converged"
    assert out["engine_success"]

    rel_excess = (chi2_engine - chi2_ref) / max(abs(chi2_ref), 1e-300)
    assert chi2_engine <= chi2_ref * (1.0 + _NO_WORSE_RTOL), (
        f"individual real-data: engine objective {chi2_engine!r} is STRICTLY WORSE "
        f"than production {chi2_ref!r} beyond the {_NO_WORSE_RTOL:.0e} keep-better "
        f"band (rel_excess={rel_excess:.3e}). On real noisy C044 data the two paths "
        "are expected to be near-tied; a larger engine excess is a real "
        "residual/scaling/layout/solver regression. Do NOT loosen this band."
    )
