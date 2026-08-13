"""Persist service: import-path equivalence + JAX-free guard."""

import subprocess
import sys
import textwrap


def _probe_import(module: str) -> int:
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        try:
            importlib.import_module({module!r})
        except BaseException:
            sys.exit(2)
        sys.exit(1 if "jax" in sys.modules else 0)
        """
    )
    return subprocess.run([sys.executable, "-c", code], check=False).returncode


def test_persist_service_is_jax_free():
    # The GUI process may import the persist service in-process; keep it JAX-free.
    assert _probe_import("xpcsjax.service.persist") == 0


def test_cli_shim_reexports_same_callable():
    from xpcsjax.cli.result_saving import save_results as cli_save
    from xpcsjax.service.persist import save_results as svc_save

    assert cli_save is svc_save


def test_save_results_rejects_bad_format(tmp_path):
    from unittest.mock import MagicMock

    import pytest

    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service.persist import save_results

    with pytest.raises(ValueError, match="Unknown output_format"):
        save_results(MagicMock(spec=OptimizationResult), tmp_path, "xml", None, None)


def test_extract_parameters_handles_zero_dim_result():
    """A 0-D result.parameters (__post_init__ permits it) must not IndexError.

    _extract_parameters used params.shape[0], which raises IndexError on a
    0-D array; .ravel().size (matching the uncertainties handling) fixes it.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from xpcsjax.service.persist import _extract_parameters

    result = MagicMock()
    result.parameters = np.array(3.0)  # 0-D
    result.uncertainties = None

    out = _extract_parameters(result, parameter_names=None)
    assert out == {"param_0": {"value": 3.0, "uncertainty": None}}


def test_json_safe_handles_zero_dim_ndarray():
    """0-D numpy arrays (scalar arrays) must coerce, not crash.

    Regression: ``np.ndarray.tolist()`` collapses a 0-D array to a bare Python
    scalar, so the old ``[_json_safe(v) for v in value.tolist()]`` iterated a
    non-iterable and raised ``TypeError: 'float' object is not iterable``. A 0-D
    array reaches ``_json_safe`` whenever a scalar numpy reduction lands in
    ``nlsq_diagnostics`` / ``streaming_diagnostics`` / ``device_info`` / cli args.
    """
    import numpy as np

    from xpcsjax.service.persist import _json_safe

    # 0-D float / int route through the scalar branches and coerce to primitives.
    assert _json_safe(np.array(3.5)) == 3.5
    assert isinstance(_json_safe(np.array(3.5)), float)
    assert _json_safe(np.array(7)) == 7
    assert isinstance(_json_safe(np.array(7)), int)

    # Non-finite 0-D mirrors the existing 1-D NaN/inf -> None contract.
    assert _json_safe(np.array(np.nan)) is None
    assert _json_safe(np.array(np.inf)) is None

    # Nested inside the dict/list trees that the result payload is built from.
    assert _json_safe({"metric": np.array(1.5)}) == {"metric": 1.5}
    assert _json_safe([np.array(2.0), 3.0]) == [2.0, 3.0]

    # 1-D arrays still expand to lists (no regression to the documented path).
    assert _json_safe(np.array([1.0, np.nan, 2.0])) == [1.0, None, 2.0]


def test_save_results_npz_replaces_offshape_uncertainties_with_nan(tmp_path):
    """Off-shape uncertainties/covariance must be written at the documented shapes.

    Regression: only an exact ``None`` triggered the NaN placeholder, but
    ``OptimizationResult.__post_init__`` also admits a 0-D scalar or an empty
    array for any parameter count (it raises only for a non-empty 1-D length
    mismatch). Those were stored verbatim next to a length-n ``parameters``
    array, breaking readers that index them 1:1 -- the same gate
    ``_extract_parameters`` already applies for the JSON writer.
    """
    import numpy as np

    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service.persist import save_results_npz

    result = OptimizationResult(
        parameters=np.array([1.0, 2.0, 3.0]),
        uncertainties=np.array(np.nan),  # 0-D placeholder: legal per __post_init__
        covariance=np.array([]),  # empty placeholder: also legal
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=3,
        execution_time=0.01,
        device_info={"device": "cpu"},
    )

    path = save_results_npz(result, tmp_path, filename="offshape.npz")

    with np.load(path, allow_pickle=False) as npz:
        assert npz["uncertainties"].shape == (3,)
        assert np.isnan(npz["uncertainties"]).all()
        assert npz["covariance"].shape == (3, 3)
        assert np.isnan(npz["covariance"]).all()

    # A correctly-shaped covariance is still stored verbatim (no regression).
    result.covariance = np.eye(3)
    path = save_results_npz(result, tmp_path, filename="inshape.npz")
    with np.load(path, allow_pickle=False) as npz:
        assert np.array_equal(npz["covariance"], np.eye(3))


def test_save_results_npz_readable_without_allow_pickle(tmp_path):
    """nlsq_result.npz must round-trip with allow_pickle=False (no SEC-1 regression).

    Regression: parameter_names/metadata_json/config_json were previously written
    with dtype=object, which numpy can only pickle-serialize -- forcing every
    reader to pass allow_pickle=True to open the file at all, the exact pattern
    data/xpcs_loader.py and data/performance_engine.py deliberately avoid.
    """
    import json

    import numpy as np

    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service.persist import save_results_npz

    result = OptimizationResult(
        parameters=np.array([1.0, 2.0]),
        uncertainties=np.array([0.1, 0.2]),
        covariance=np.eye(2),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=3,
        execution_time=0.01,
        device_info={"device": "cpu"},
    )

    class _FakeConfigManager:
        def get_active_parameters(self):
            return ["D0", "alpha"]

    path = save_results_npz(result, tmp_path, config_manager=_FakeConfigManager())

    with np.load(path, allow_pickle=False) as npz:
        assert list(npz["parameter_names"]) == ["D0", "alpha"]
        assert npz["parameter_names"].dtype.kind == "U"  # not object -- true string dtype
        assert "success" in str(npz["metadata_json"])
        assert npz["metadata_json"].dtype.kind == "U"
        assert npz["config_json"].dtype.kind == "U"
        assert json.loads(str(npz["config_json"]))  # round-trips as real JSON, not a pickle blob
        assert npz["parameters"].dtype == np.float64


def test_resolve_parameter_names_prefers_nlsq_diagnostics():
    """Heterodyne fits label from result.nlsq_diagnostics, not the config manager.

    Regression: the length-mismatch warning fired on every scaled fit because
    ConfigManager.get_active_parameters() is physics-only by design. Heterodyne
    dispatch attaches its own (scaling + physics) label list to
    result.nlsq_diagnostics["parameter_names"] -- that must win over the
    shorter, physics-only config-manager list.
    """
    from unittest.mock import MagicMock

    from xpcsjax.service.persist import _resolve_parameter_names

    result = MagicMock()
    result.nlsq_diagnostics = {"parameter_names": ["contrast_0", "offset_0", "D0_ref", "alpha_ref"]}

    class _FakeConfigManager:
        def get_active_parameters(self):
            return ["D0_ref", "alpha_ref"]  # physics-only -- shorter, must lose

    names = _resolve_parameter_names(_FakeConfigManager(), result)

    assert names == ["contrast_0", "offset_0", "D0_ref", "alpha_ref"]


def test_resolve_parameter_names_synthesizes_scaling_head_for_homodyne():
    """Homodyne fits (no nlsq_diagnostics parameter_names) get names reconstructed
    from the vector-length delta against the physics-only config list, instead
    of hitting the length-mismatch warning and falling back to param_0, param_1.

    This is the static_isotropic shape from the original bug report: 3 physics
    params + a single-angle (contrast, offset) scaling head = 5 total.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from xpcsjax.service.persist import _resolve_parameter_names

    result = MagicMock()
    result.nlsq_diagnostics = None  # homodyne: never populated
    result.parameters = np.zeros(5)  # 3 physics + 1-angle (contrast, offset)

    class _FakeConfigManager:
        def get_active_parameters(self):
            return ["D0", "alpha", "D_offset"]

    names = _resolve_parameter_names(_FakeConfigManager(), result)

    assert names == ["contrast_0", "offset_0", "D0", "alpha", "D_offset"]


def test_resolve_parameter_names_synthesizes_multi_angle_scaling_head():
    """Multi-angle 'individual' per-angle mode: N (contrast, offset) pairs."""
    from unittest.mock import MagicMock

    import numpy as np

    from xpcsjax.service.persist import _resolve_parameter_names

    result = MagicMock()
    result.nlsq_diagnostics = {}
    result.parameters = np.zeros(7)  # 3 physics + 2-angle scaling head (4)

    class _FakeConfigManager:
        def get_active_parameters(self):
            return ["D0", "alpha", "D_offset"]

    names = _resolve_parameter_names(_FakeConfigManager(), result)

    assert names == ["contrast_0", "contrast_1", "offset_0", "offset_1", "D0", "alpha", "D_offset"]


def test_save_results_json_labels_homodyne_scaling_params(tmp_path):
    """End-to-end: a homodyne-shaped result no longer degrades to param_0..N,
    and the JSON header's parameter_names agrees with the parameters block
    (both come from the same resolved list).
    """
    import json

    import numpy as np

    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service.persist import save_results_json

    result = OptimizationResult(
        parameters=np.array([0.05, 1.0, 16834.9, -1.57, 3.03]),
        uncertainties=None,
        covariance=None,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=3,
        execution_time=0.01,
        device_info={"device": "cpu"},
    )

    class _FakeConfigManager:
        config = {"analysis_mode": "static_isotropic"}
        config_file = "config.yaml"

        def get_active_parameters(self):
            return ["D0", "alpha", "D_offset"]

    path = save_results_json(result, tmp_path, config_manager=_FakeConfigManager())
    payload = json.loads(path.read_text())

    expected = ["contrast_0", "offset_0", "D0", "alpha", "D_offset"]
    assert list(payload["parameters"].keys()) == expected
    assert payload["config"]["parameter_names"] == expected  # header agrees with body
    assert "param_0" not in payload["parameters"]


def _write_fitted_c2_npz(path, *, c2_exp=None, mtime=None):
    import os

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if c2_exp is None:
        c2_exp = np.full((2, 3, 3), 1.0)
    np.savez(
        path,
        c2_exp=c2_exp,
        c2_fitted=c2_exp * 0.9,
        residuals=c2_exp * 0.1,
        t1=np.arange(3, dtype=float),
        t2=np.arange(3, dtype=float),
        phi_angles=np.array([0.0, 45.0]),
        # Deliberately-excluded keys, present in the real writer's output too.
        params=np.array([1.0, 2.0]),
        contrast=np.float64(1.0),
        offset=np.float64(0.0),
        q=np.float64(0.01),
        reduced_chi_squared=np.float64(1.0),
    )
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _write_primary_npz(path, *, mtime=None):
    import os

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, parameters=np.array([1.0, 2.0]), reduced_chi_squared=np.float64(2.0))
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_merge_fitted_c2_happy_path(tmp_path):
    import numpy as np

    from xpcsjax.service.persist import merge_fitted_c2

    npz_path = tmp_path / "nlsq_result.npz"
    fitted_npz = tmp_path / "plots" / "simulated_data" / "c2_fitted_data.npz"
    c2_exp = np.full((2, 3, 3), 1.0)
    _write_primary_npz(npz_path)
    _write_fitted_c2_npz(fitted_npz, c2_exp=c2_exp)

    assert merge_fitted_c2(npz_path, fitted_npz) is True

    merged = np.load(npz_path, allow_pickle=False)
    assert set(merged.files) >= {
        "parameters",
        "reduced_chi_squared",
        "c2_exp",
        "c2_fitted",
        "residuals",
        "t1",
        "t2",
        "phi_angles",
    }
    assert np.array_equal(merged["c2_exp"], c2_exp)
    # Original primary-npz value must win on key collision (reduced_chi_squared
    # exists in both source files with different values: 2.0 here vs 1.0 in
    # the fitted-c2 sidecar).
    assert merged["reduced_chi_squared"] == 2.0
    # The fitted-c2 sidecar's own params/contrast/offset/q are deliberately
    # NOT pulled in -- they duplicate/shadow different-shaped primary-npz data.
    assert "params" not in merged.files
    assert "contrast" not in merged.files


def test_merge_fitted_c2_missing_fitted_npz_is_noop(tmp_path):
    from xpcsjax.service.persist import merge_fitted_c2

    npz_path = tmp_path / "nlsq_result.npz"
    _write_primary_npz(npz_path)
    fitted_npz = tmp_path / "plots" / "simulated_data" / "c2_fitted_data.npz"

    assert merge_fitted_c2(npz_path, fitted_npz) is False
    # Primary NPZ is untouched.
    import numpy as np

    assert set(np.load(npz_path, allow_pickle=False).files) == {
        "parameters",
        "reduced_chi_squared",
    }


def test_merge_fitted_c2_missing_primary_npz_is_noop(tmp_path):
    from xpcsjax.service.persist import merge_fitted_c2

    npz_path = tmp_path / "nlsq_result.npz"
    fitted_npz = tmp_path / "plots" / "simulated_data" / "c2_fitted_data.npz"
    _write_fitted_c2_npz(fitted_npz)

    assert merge_fitted_c2(npz_path, fitted_npz) is False
    assert not npz_path.exists()


def test_merge_fitted_c2_rejects_stale_fitted_npz(tmp_path):
    """A fitted-c2 NPZ older than the primary NPZ is a leftover from a
    PREVIOUS run in the same output directory -- merging it would silently
    pair this run's parameters with another run's c2 arrays.
    """
    import time

    import numpy as np

    from xpcsjax.service.persist import merge_fitted_c2

    npz_path = tmp_path / "nlsq_result.npz"
    fitted_npz = tmp_path / "plots" / "simulated_data" / "c2_fitted_data.npz"
    now = time.time()
    # Fitted-c2 written well BEFORE the primary npz -- e.g. this run's
    # plotting failed silently, leaving a prior run's sidecar in place while
    # save_results_npz still overwrote the primary npz fresh.
    _write_fitted_c2_npz(fitted_npz, mtime=now - 3600)
    _write_primary_npz(npz_path, mtime=now)

    assert merge_fitted_c2(npz_path, fitted_npz) is False
    merged = np.load(npz_path, allow_pickle=False)
    assert "c2_exp" not in merged.files


def test_merge_fitted_c2_accepts_same_or_newer_fitted_npz(tmp_path):
    """Not stale: fitted-c2 written at/after the primary npz (the real-world
    ordering, since save_results_npz always writes before plotting runs).
    """
    import time

    import numpy as np

    from xpcsjax.service.persist import merge_fitted_c2

    npz_path = tmp_path / "nlsq_result.npz"
    fitted_npz = tmp_path / "plots" / "simulated_data" / "c2_fitted_data.npz"
    now = time.time()
    _write_primary_npz(npz_path, mtime=now)
    _write_fitted_c2_npz(fitted_npz, mtime=now + 5)

    assert merge_fitted_c2(npz_path, fitted_npz) is True
    merged = np.load(npz_path, allow_pickle=False)
    assert "c2_exp" in merged.files
