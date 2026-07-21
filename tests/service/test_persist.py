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
