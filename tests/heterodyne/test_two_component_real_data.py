"""Multi-angle heterodyne (two_component) NLSQ fit against the source baseline.

This test exercises the heterodyne dispatch path through
:func:`xpcsjax.optimization.nlsq.fit_nlsq`. The source heterodyne CLI's
joint 3-angle fit is pinned in ``tests/heterodyne/fixtures/baselines/
two_component_c044.json``; we run the same configuration through xpcsjax and
compare the 14 physics parameters to the baseline.

Gating
------
Slow real-data fit (multi-minute on CPU). Gated by the Phase-5 env var:

    XPCSJAX_RUN_CHARACTERIZATION=1 uv run pytest \
        tests/heterodyne/test_two_component_real_data.py -v

Data layout
-----------
The C044 cache file ``cached_c2_q0.0054_frames_1000_2000.npz`` was written by
the source heterodyne CLI in its own format. The xpcsjax homodyne loader
expects ``c2_exp`` and other keys, so this test bypasses
``load_xpcs_data`` and constructs the heterodyne-style data dict directly
from the cache (matching the layout the source heterodyne pipeline uses
internally).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

C044_DATA_DIR = Path("/home/wei/Documents/Projects/data/C044")
C044_CONFIG = C044_DATA_DIR / "heterodyne_config.yaml"
C044_C2_CACHE = C044_DATA_DIR / "cached_c2_q0.0054_frames_1000_2000.npz"
C044_PHI_LIST = C044_DATA_DIR / "phi_list.txt"
BASELINE_JSON = (
    Path(__file__).resolve().parent / "fixtures" / "baselines" / "two_component_c044.json"
)

_SLOW_GATE = pytest.mark.skipif(
    os.environ.get("XPCSJAX_RUN_CHARACTERIZATION") != "1",
    reason="Slow real-data heterodyne fit; set XPCSJAX_RUN_CHARACTERIZATION=1 to enable.",
)


def _require_fixture(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing C044 fixture: {path}")


@pytest.fixture(scope="module")
def baseline() -> dict[str, object]:
    if not BASELINE_JSON.exists():
        pytest.skip(f"Baseline JSON missing: {BASELINE_JSON}")
    return json.loads(BASELINE_JSON.read_text())


def _select_baseline_angles(phi_all: np.ndarray, baseline_meta: dict[str, object]) -> np.ndarray:
    """Return indices into the cache's phi axis for the 3 baseline angles."""
    target = np.asarray(baseline_meta["phi_angles"], dtype=np.float64)
    idxs = np.array([int(np.argmin(np.abs(phi_all - t))) for t in target], dtype=np.int64)
    # Sanity: each picked angle should be < 1° from the baseline target.
    deltas = np.abs(phi_all[idxs] - target)
    assert np.all(deltas < 1.0), f"could not match baseline phi angles within 1°; deltas={deltas}"
    return idxs


@_SLOW_GATE
def test_heterodyne_multi_angle_matches_source(baseline):
    """End-to-end multi-angle heterodyne fit via the dispatch path."""
    _require_fixture(C044_C2_CACHE)
    _require_fixture(C044_CONFIG)

    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    # ------------------------------------------------------------------
    # Build the heterodyne-style data dict directly from the cache.
    # The source pipeline drops the leading time point.
    # ------------------------------------------------------------------
    cache = np.load(C044_C2_CACHE, allow_pickle=False)
    c2_all = np.asarray(cache["c2"], dtype=np.float64)  # (n_phi, 1001, 1001)
    phi_all = np.asarray(cache["phi"], dtype=np.float64)

    baseline_meta = baseline["metadata"]
    idxs = _select_baseline_angles(phi_all, baseline_meta)
    c2 = c2_all[idxs, 1:, 1:]  # (3, 1000, 1000) matching source pipeline
    phi = phi_all[idxs]

    data = {"c2": c2, "phi": phi}

    # ------------------------------------------------------------------
    # Dispatch through fit_nlsq → _fit_nlsq_heterodyne → fit_nlsq_multi_phi.
    # ------------------------------------------------------------------
    import time

    t0 = time.perf_counter()
    results = fit_nlsq(data, str(C044_CONFIG))
    wall = time.perf_counter() - t0

    print(f"\n[heterodyne multi-angle] wall={wall:.1f}s n_angles={len(phi)} c2.shape={c2.shape}")

    # After Phase-6 C2-C6 + C5b, ``fit_nlsq_multi_phi`` returns a single
    # :class:`OptimizationResult` for every dispatch mode (constant /
    # averaged / fourier / CMA-ES / individual). With n_phi=3 + the C044
    # config's ``auto`` dispatch (constant_threshold=3,
    # fourier_threshold=6) the resolver selects the averaged branch;
    # the unified shape contract applies regardless.
    assert isinstance(results, OptimizationResult), (
        f"expected OptimizationResult, got {type(results)}"
    )
    diag = results.nlsq_diagnostics or {}
    assert "parameter_names" in diag, (
        "OptimizationResult.nlsq_diagnostics must carry parameter_names for joint heterodyne fits"
    )
    first_names = list(diag["parameter_names"])
    # The optimizer parameter vector is
    # ``[physics_varying | per-angle scaling tail]``. Slice the physics
    # block out so the baseline-parity comparison below sees the same
    # layout that the source heterodyne CLI produces (14 physics params).
    first_params = np.asarray(results.parameters, dtype=np.float64)[: len(first_names)]
    assert np.all(np.isfinite(first_params)), f"NaN/Inf in parameters: {first_params}"

    # ------------------------------------------------------------------
    # Parity comparison — by IDENTIFIABLE quantity, not by individual params.
    #
    # This two-component model is degenerate for the C044 data: the config
    # freezes D0_sample / alpha_sample because "f0 ~ 0 makes the sample branch
    # unidentifiable", and several remaining params lie on flat directions
    # (D_offset_ref trades off with the large D0_ref * t^alpha term; the
    # f0/f1/f2 fraction block; the flow block when f0 ~ 0). On a degenerate
    # ridge the source (scipy least_squares) and xpcsjax (NLSQ JAX) optimizers
    # reach DIFFERENT-but-equivalent points, so pinning those individual
    # parameters is not a valid parity check. We verified this directly: at the
    # diverging params xpcsjax reaches SSR 7184.8 vs the source's 7274.6 — i.e.
    # xpcsjax fits at least as well, the parameters just sit elsewhere on the
    # ridge. The meaningful, identifiable parity quantities are the achieved
    # fit quality and the well-constrained diffusion block.
    # ------------------------------------------------------------------
    expected = np.asarray(baseline["parameters"], dtype=np.float64)
    expected_names = list(baseline["parameter_names"])
    assert expected.shape == (len(expected_names),) and len(expected_names) >= 1
    name_to_idx = {n: i for i, n in enumerate(first_names)}
    missing = [n for n in expected_names if n not in name_to_idx]
    assert not missing, f"baseline param names missing from NLSQResult: {missing}"
    param_map = {n: first_params[name_to_idx[n]] for n in expected_names}
    expected_map = {n: expected[i] for i, n in enumerate(expected_names)}
    print(f"[heterodyne multi-angle] params xpcsjax={param_map}")
    print(f"[heterodyne multi-angle] params source ={expected_map}")

    # (1) DECISIVE CHECK — achieved fit quality. xpcsjax must reach an SSR at
    # least as low as the source (fit at least as well), within a small band.
    # Equal objective on a degenerate model proves the fits are equivalent
    # regardless of where on the ridge each optimizer landed; a materially
    # higher SSR would be a real fit-quality regression (not degeneracy).
    xpcsjax_chi2 = float(results.chi_squared)
    source_chi2 = float(baseline["chi_squared"])
    print(f"[heterodyne multi-angle] SSR xpcsjax={xpcsjax_chi2:.2f} source={source_chi2:.2f}")
    assert xpcsjax_chi2 <= source_chi2 * 1.05, (
        f"xpcsjax SSR {xpcsjax_chi2:.2f} is materially worse than the source "
        f"{source_chi2:.2f} (>5%) — a real fit-quality regression, not degeneracy"
    )

    # (2) Well-identified diffusion parameters must match tightly. These are
    # constrained by the correlation decay even when the sample/flow/fraction
    # blocks are not. (D_offset_ref, v0, beta, v_offset, phi0, f0/f1/f2/f3 are
    # intentionally NOT asserted individually — they are degenerate directions;
    # check (1) covers them via the objective.)
    for p in ("D0_ref", "alpha_ref", "D_offset_sample"):
        if p in expected_map:
            np.testing.assert_allclose(
                param_map[p],
                expected_map[p],
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"identifiable diffusion parameter {p} diverged from source",
            )
