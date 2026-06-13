"""Phase 1+2 — C044 in-memory scaling-first no-worse-SSR oracle (maintainer-gated).

Mirrors the gate of tests/parity/test_engine_heterodyne_realdata_c044.py: the
scaling-first re-order of the heterodyne in-memory path (native + escapes +
engine route) must be NO-WORSE on real C044 data for all three modes —
``constant`` strict (1e-6), ``individual``/``averaged`` no-worse within 1e-3 — vs
a captured pre-re-order baseline SSR. Off by default; never runs in CI / fresh
clones.

The committed baseline (``fixtures/c044_inmemory_ssr_baseline.json``) was captured
ONCE at the pre-re-order commit ``47e5323`` (last commit before the native
scaling-first re-order ``65e3f5a``). This test then asserts the CURRENT
scaling-first SSR is no-worse than that baseline — the empirical basin-neutrality
gate for the re-order. Regenerate with::

    python scripts/realdata_engine_fit_parity_c044.py --emit-baseline --n-t 48 --nfev 400

(on the CURRENT branch that emits the NEW SSR; the committed JSON is the
pre-re-order one and must NOT be overwritten from a post-re-order checkout).

``averaged`` is not a user-facing ``per_angle_mode`` token; it is the variant
``auto`` resolves to at C044's ``n_phi >= 3``, so it is driven via production
``auto`` (same mapping the baseline / ``--emit-baseline`` use).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

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
# Captured at the START of Phase 1+2 (pre-re-order, commit 47e5323) by
# scripts/realdata_engine_fit_parity_c044.py --emit-baseline; committed alongside.
_BASELINE = Path(__file__).parent / "fixtures" / "c044_inmemory_ssr_baseline.json"

_N_T = 48
_NFEV = 400
_NO_WORSE_RTOL = 1e-3
_STRICT_RTOL = 1e-6


@pytest.mark.skipif(not _GATE_OK, reason=_SKIP_REASON)
@pytest.mark.skipif(not _BASELINE.is_file(), reason="pre-re-order SSR baseline not captured")
@pytest.mark.parametrize("mode", ["constant", "individual", "averaged"])
def test_c044_inmemory_scaling_first_no_worse(mode):
    # Reuse the parity script's loader + mode mapping so no model/engine
    # construction is duplicated (lifted exactly like the sibling C044 oracle).
    from tests.parity.test_engine_heterodyne_realdata_c044 import _load_helpers

    helpers = _load_helpers()
    model, c2, phi = helpers.load_c044(str(_C044_CONFIG), n_t=_N_T)

    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    production_mode = helpers._BASELINE_MODE_TO_PRODUCTION[mode]
    cfg = NLSQConfig(per_angle_mode=production_mode, max_nfev=_NFEV)
    result = fit_nlsq_multi_phi(model, c2, list(phi), cfg, None)
    ssr_now = float(result.chi_squared)

    baseline = json.loads(_BASELINE.read_text())[mode]
    ssr_base = float(baseline["chi_squared"])
    assert np.isfinite(ssr_now) and np.isfinite(ssr_base) and ssr_base > 0.0

    rtol = _STRICT_RTOL if mode == "constant" else _NO_WORSE_RTOL
    assert ssr_now <= ssr_base * (1.0 + rtol), (
        f"{mode}: scaling-first SSR {ssr_now:.6e} worse than pre-re-order baseline "
        f"{ssr_base:.6e} beyond rtol={rtol}. This is a REAL basin regression from "
        "the scaling-first re-order — diagnose it; do NOT loosen the tolerance."
    )
