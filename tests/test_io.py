"""Tests for xpcsjax/io module: json_utils and nlsq_writers."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from xpcsjax.io.json_utils import _JSON_ARRAY_SIZE_LIMIT, json_safe, json_serializer
from xpcsjax.io.nlsq_writers import save_nlsq_json_files, save_nlsq_npz_file

# ---------------------------------------------------------------------------
# json_safe
# ---------------------------------------------------------------------------


class TestJsonSafeFloatSanitization:
    def test_nan_becomes_none(self) -> None:
        assert json_safe(float("nan")) is None

    def test_pos_inf_becomes_string(self) -> None:
        assert json_safe(float("inf")) == "Infinity"

    def test_neg_inf_becomes_string(self) -> None:
        assert json_safe(float("-inf")) == "-Infinity"

    def test_finite_float_unchanged(self) -> None:
        assert json_safe(3.14) == pytest.approx(3.14)

    def test_numpy_nan_float64_becomes_none(self) -> None:
        assert json_safe(np.float64("nan")) is None

    def test_numpy_inf_float64_becomes_string(self) -> None:
        assert json_safe(np.float64("inf")) == "Infinity"


class TestJsonSafeNumpyTypes:
    def test_int_array(self) -> None:
        assert json_safe(np.array([1, 2, 3])) == [1, 2, 3]

    def test_float_array_with_nan(self) -> None:
        result = json_safe(np.array([1.0, float("nan"), 2.0]))
        assert result == [pytest.approx(1.0), None, pytest.approx(2.0)]

    def test_numpy_integer_scalar(self) -> None:
        assert json_safe(np.int64(42)) == 42

    def test_numpy_bool(self) -> None:
        assert json_safe(np.bool_(True)) is True
        assert json_safe(np.bool_(False)) is False


class TestJsonSafeContainers:
    def test_nested_dict(self) -> None:
        d = {"a": float("nan"), "b": {"c": float("inf")}}
        result = json_safe(d)
        assert result == {"a": None, "b": {"c": "Infinity"}}

    def test_list_with_nan(self) -> None:
        result = json_safe([1.0, float("nan"), float("inf")])
        assert result == [pytest.approx(1.0), None, "Infinity"]

    def test_tuple_converted_to_list(self) -> None:
        result = json_safe((1, 2, 3))
        assert result == [1, 2, 3]


class TestJsonSafeEdgeCases:
    def test_complex_number_splits_to_dict(self) -> None:
        result = json_safe(complex(1.5, -2.5))
        assert result == {"real": pytest.approx(1.5), "imag": pytest.approx(-2.5)}

    def test_complex_with_nan_imag(self) -> None:
        result = json_safe(complex(1.0, float("nan")))
        assert result["real"] == pytest.approx(1.0)
        assert result["imag"] is None

    def test_path_converts_to_str(self) -> None:
        p = Path("/tmp/xpcsjax/output")
        assert json_safe(p) == "/tmp/xpcsjax/output"

    def test_plain_int_passes_through(self) -> None:
        assert json_safe(42) == 42

    def test_string_passes_through(self) -> None:
        assert json_safe("hello") == "hello"

    def test_large_array_raises(self) -> None:
        big = np.ones(_JSON_ARRAY_SIZE_LIMIT + 1)
        with pytest.raises(ValueError, match="too large to embed in JSON"):
            json_safe(big)

    def test_array_at_limit_is_accepted(self) -> None:
        ok = np.zeros(_JSON_ARRAY_SIZE_LIMIT)
        result = json_safe(ok)
        assert len(result) == _JSON_ARRAY_SIZE_LIMIT


class TestJsonSafeRoundTrip:
    """Verify json_safe output is always valid JSON (no NaN/Inf tokens)."""

    def test_dict_with_nan_roundtrips(self) -> None:
        d = {"value": float("nan"), "arr": np.array([float("nan"), 1.0])}
        safe = json_safe(d)
        dumped = json.dumps(safe)
        loaded = json.loads(dumped)
        assert loaded["value"] is None
        assert loaded["arr"][0] is None
        assert loaded["arr"][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# json_serializer
# ---------------------------------------------------------------------------


class TestJsonSerializer:
    def test_numpy_array(self) -> None:
        result = json_serializer(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_numpy_integer(self) -> None:
        assert json_serializer(np.int64(5)) == 5
        assert isinstance(json_serializer(np.int64(5)), int)

    def test_numpy_floating_nan(self) -> None:
        assert json_serializer(np.float64("nan")) is None

    def test_numpy_bool(self) -> None:
        assert json_serializer(np.bool_(True)) is True

    def test_plain_int_returns_int_not_str(self) -> None:
        result = json_serializer(42)
        assert result == 42
        assert isinstance(result, int)

    def test_plain_float_nan(self) -> None:
        assert json_serializer(float("nan")) is None

    def test_complex_splits(self) -> None:
        result = json_serializer(complex(1, 2))
        assert result == {"real": 1.0, "imag": pytest.approx(2.0)}

    def test_unknown_type_falls_back_to_str(self) -> None:
        class MyObj:
            def __repr__(self) -> str:
                return "MyObj"

        result = json_serializer(MyObj())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# save_nlsq_json_files — critical: plain float NaN must not produce NaN token
# ---------------------------------------------------------------------------


class TestSaveNlsqJsonFiles:
    def _make_dicts(self) -> tuple[dict, dict, dict]:
        param = {
            "gamma": {"value": float("nan"), "uncertainty": float("inf")},
            "D": {"value": 1.23, "uncertainty": 0.01},
        }
        analysis = {"reduced_chi_squared": float("nan"), "method": "nlsq"}
        convergence = {"status": "converged", "iterations": 42}
        return param, analysis, convergence

    def test_no_nan_token_in_output(self) -> None:
        param, analysis, convergence = self._make_dicts()
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_json_files(param, analysis, convergence, Path(tmp))
            for fname in ("parameters.json", "analysis_results_nlsq.json"):
                text = (Path(tmp) / fname).read_text()
                assert "NaN" not in text, f"Invalid JSON NaN token in {fname}"
                assert ": Infinity" not in text or '"Infinity"' in text

    def test_output_is_valid_json(self) -> None:
        param, analysis, convergence = self._make_dicts()
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_json_files(param, analysis, convergence, Path(tmp))
            for fname in (
                "parameters.json",
                "analysis_results_nlsq.json",
                "convergence_metrics.json",
            ):
                text = (Path(tmp) / fname).read_text()
                json.loads(text)  # raises if invalid

    def test_nan_in_param_becomes_null(self) -> None:
        param, analysis, convergence = self._make_dicts()
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_json_files(param, analysis, convergence, Path(tmp))
            loaded = json.loads((Path(tmp) / "parameters.json").read_text())
            assert loaded["gamma"]["value"] is None

    def test_creates_directory_if_missing(self) -> None:
        param, analysis, convergence = self._make_dicts()
        with tempfile.TemporaryDirectory() as tmp:
            subdir = Path(tmp) / "nested" / "output"
            save_nlsq_json_files(param, analysis, convergence, subdir)
            assert (subdir / "parameters.json").exists()


# ---------------------------------------------------------------------------
# save_nlsq_npz_file
# ---------------------------------------------------------------------------


def _make_npz_arrays(n_angles: int = 3, n_t: int = 5) -> dict:
    """Build minimal valid arrays for save_nlsq_npz_file."""
    shape = (n_angles, n_t, n_t)
    rng = np.random.default_rng(0)
    return dict(
        phi_angles=rng.uniform(0, 360, n_angles),
        c2_exp=rng.uniform(0, 1, shape),
        c2_raw=rng.uniform(0, 1, shape),
        c2_scaled=rng.uniform(0, 1, shape),
        c2_solver=None,
        per_angle_scaling=rng.uniform(0, 1, (n_angles, 2)),
        per_angle_scaling_solver=rng.uniform(0, 1, (n_angles, 2)),
        residuals=rng.uniform(-0.1, 0.1, shape),
        residuals_norm=rng.uniform(-1, 1, shape),
        t1=np.linspace(1e-3, 1.0, n_t),
        t2=np.linspace(1e-3, 1.0, n_t),
        q=0.05,
    )


class TestSaveNlsqNpzFile:
    def test_basic_write_loads_back(self) -> None:
        arrays = _make_npz_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_npz_file(**arrays, output_dir=Path(tmp))
            npz = np.load(Path(tmp) / "fitted_data.npz")
            assert "phi_angles" in npz
            assert "c2_exp" in npz
            assert "q" in npz

    def test_array_count_without_solver(self) -> None:
        arrays = _make_npz_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_npz_file(**arrays, output_dir=Path(tmp))
            npz = np.load(Path(tmp) / "fitted_data.npz")
            assert len(npz.files) == 11

    def test_array_count_with_solver(self) -> None:
        arrays = _make_npz_arrays()
        arrays["c2_solver"] = np.zeros((3, 5, 5))
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_npz_file(**arrays, output_dir=Path(tmp))
            npz = np.load(Path(tmp) / "fitted_data.npz")
            assert len(npz.files) == 12

    def test_shape_mismatch_raises(self) -> None:
        arrays = _make_npz_arrays()
        arrays["c2_exp"] = np.zeros((4, 5, 5))  # wrong n_angles
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="c2_exp.shape"):
                save_nlsq_npz_file(**arrays, output_dir=Path(tmp))

    def test_jax_array_accepted(self) -> None:
        """JAX arrays must be coerced without error."""
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not available")
        arrays = _make_npz_arrays()
        arrays["c2_exp"] = jnp.array(arrays["c2_exp"])
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_npz_file(**arrays, output_dir=Path(tmp))
            npz = np.load(Path(tmp) / "fitted_data.npz")
            assert isinstance(npz["c2_exp"], np.ndarray)

    def test_creates_directory_if_missing(self) -> None:
        arrays = _make_npz_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            subdir = Path(tmp) / "new" / "dir"
            save_nlsq_npz_file(**arrays, output_dir=subdir)
            assert (subdir / "fitted_data.npz").exists()

    def test_q_wrapped_as_array(self) -> None:
        arrays = _make_npz_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            save_nlsq_npz_file(**arrays, output_dir=Path(tmp))
            npz = np.load(Path(tmp) / "fitted_data.npz")
            assert npz["q"].shape == (1,)
            assert math.isclose(float(npz["q"][0]), arrays["q"])
