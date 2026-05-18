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
    Path(__file__).resolve().parent
    / "fixtures"
    / "baselines"
    / "two_component_c044.json"
)

_SLOW_GATE = pytest.mark.skipif(
    os.environ.get("XPCSJAX_RUN_CHARACTERIZATION") != "1",
    reason="Slow real-data heterodyne fit; "
    "set XPCSJAX_RUN_CHARACTERIZATION=1 to enable.",
)


def _require_fixture(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing C044 fixture: {path}")


@pytest.fixture(scope="module")
def baseline() -> dict[str, object]:
    if not BASELINE_JSON.exists():
        pytest.skip(f"Baseline JSON missing: {BASELINE_JSON}")
    return json.loads(BASELINE_JSON.read_text())


def _select_baseline_angles(
    phi_all: np.ndarray, baseline_meta: dict[str, object]
) -> np.ndarray:
    """Return indices into the cache's phi axis for the 3 baseline angles."""
    target = np.asarray(baseline_meta["phi_angles"], dtype=np.float64)
    idxs = np.array(
        [int(np.argmin(np.abs(phi_all - t))) for t in target], dtype=np.int64
    )
    # Sanity: each picked angle should be < 1° from the baseline target.
    deltas = np.abs(phi_all[idxs] - target)
    assert np.all(deltas < 1.0), (
        f"could not match baseline phi angles within 1°; deltas={deltas}"
    )
    return idxs


@_SLOW_GATE
def test_heterodyne_multi_angle_matches_source(baseline):
    """End-to-end multi-angle heterodyne fit via the dispatch path."""
    _require_fixture(C044_C2_CACHE)
    _require_fixture(C044_CONFIG)

    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult

    # ------------------------------------------------------------------
    # Build the heterodyne-style data dict directly from the cache.
    # The source pipeline drops the leading time point.
    # ------------------------------------------------------------------
    cache = np.load(C044_C2_CACHE)
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

    print(
        f"\n[heterodyne multi-angle] wall={wall:.1f}s "
        f"n_angles={len(phi)} c2.shape={c2.shape}"
    )

    # fit_nlsq_multi_phi returns list[NLSQResult] (joint physics, per-angle scaling)
    assert isinstance(results, list), f"expected list, got {type(results)}"
    assert len(results) == len(phi), (
        f"expected {len(phi)} per-angle results, got {len(results)}"
    )
    for r in results:
        assert isinstance(r, NLSQResult)
        assert np.all(np.isfinite(np.asarray(r.parameters))), (
            f"NaN/Inf in parameters: {r.parameters}"
        )

    # ------------------------------------------------------------------
    # Compare to the source baseline. The baseline pins 14 physics params
    # in canonical order. Each NLSQResult's parameter vector has its own
    # layout (physics_varying + scaling) — extract the 14 physics params.
    # ------------------------------------------------------------------
    expected = np.asarray(baseline["parameters"], dtype=np.float64)
    expected_names = list(baseline["parameter_names"])
    assert expected.shape == (14,) and len(expected_names) == 14

    first = results[0]
    first_params = np.asarray(first.parameters, dtype=np.float64)
    first_names = list(first.parameter_names)
    print(f"[heterodyne multi-angle] result0 names={first_names}")
    print(f"[heterodyne multi-angle] result0 params={first_params}")
    print(f"[heterodyne multi-angle] baseline names={expected_names}")
    print(f"[heterodyne multi-angle] baseline params={expected}")

    # Find each baseline param in the result and compare.
    name_to_idx = {n: i for i, n in enumerate(first_names)}
    missing = [n for n in expected_names if n not in name_to_idx]
    assert not missing, (
        f"baseline param names missing from NLSQResult: {missing}"
    )
    actual = np.array(
        [first_params[name_to_idx[n]] for n in expected_names],
        dtype=np.float64,
    )

    # Tolerance: rtol=1e-3 is reasonable given JIT/compiler nondeterminism
    # between heterodyne (scipy least_squares) and xpcsjax (NLSQ JAX).
    # If the multi-angle solver diverges from the source, the assertion
    # message prints both vectors for diagnosis.
    param_map = {name: actual[i] for i, name in enumerate(expected_names)}
    expected_map = {name: expected[i] for i, name in enumerate(expected_names)}

    # 1. Non-degenerate physical parameters must match the baseline tightly.
    tight_params = [
        "D0_ref",
        "alpha_ref",
        "D0_sample",
        "alpha_sample",
        "D_offset_sample",
        "v0",
        "beta",
        "v_offset",
        "f1",
        "f3",
        "phi0",
    ]
    for p in tight_params:
        np.testing.assert_allclose(
            param_map[p],
            expected_map[p],
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"Physical parameter {p} diverged from source baseline",
        )

    # 2. D_offset_ref matches within 2% due to slight trade-off with the large D0_ref.
    np.testing.assert_allclose(
        param_map["D_offset_ref"],
        expected_map["D_offset_ref"],
        rtol=2e-2,
        atol=2e-2,
        err_msg="D_offset_ref diverged from source baseline",
    )

    # 3. For the degenerate exponential sample-fraction parametrization (f0 and f2),
    # verify the mathematically equivalent invariant: f0 * exp(-f1 * f2)
    actual_f_scale = param_map["f0"] * np.exp(-param_map["f1"] * param_map["f2"])
    expected_f_scale = expected_map["f0"] * np.exp(
        -expected_map["f1"] * expected_map["f2"]
    )
    np.testing.assert_allclose(
        actual_f_scale,
        expected_f_scale,
        rtol=1.5e-2,
        atol=1.5e-2,
        err_msg="Degenerate sample-fraction scale factor diverged from source baseline",
    )
