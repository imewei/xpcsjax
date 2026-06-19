"""Tests for the JAX-free viz bundle (writer-shape + loader)."""

import numpy as np

from xpcsjax.gui.viz_bundle import VizBundle, load_viz_bundle


def _write_fitted(result_dir, **arrays):
    fitted_dir = result_dir / "plots" / "simulated_data"
    fitted_dir.mkdir(parents=True, exist_ok=True)
    np.savez(fitted_dir / "c2_fitted_data.npz", **arrays)


def test_load_missing_returns_none(tmp_path):
    assert load_viz_bundle(tmp_path) is None  # no fitted artifact


def test_load_full_bundle_from_fitted_artifact(tmp_path):
    _write_fitted(
        tmp_path,
        c2_exp=np.ones((2, 8, 8)),
        c2_fitted=np.full((2, 8, 8), 0.25),
        residuals=np.full((2, 8, 8), 0.75),
        t1=np.arange(8.0),
        t2=np.arange(8.0),
        phi_angles=np.array([0.0, 45.0]),
    )
    b = load_viz_bundle(tmp_path)
    assert isinstance(b, VizBundle)
    assert b.exp_c2.shape == (2, 8, 8)
    assert b.model_c2 is not None and b.model_c2.shape == (2, 8, 8)
    assert np.allclose(b.residuals, 0.75)  # read directly from the stored key
    assert b.phi_angles.tolist() == [0.0, 45.0]


def test_exp_only_when_no_fitted_surface(tmp_path):
    # artifact present but missing c2_fitted -> exp-only views (model/residuals None)
    _write_fitted(tmp_path, c2_exp=np.ones((2, 8, 8)))
    b = load_viz_bundle(tmp_path)
    assert b is not None and b.exp_c2.shape == (2, 8, 8)
    assert b.model_c2 is None and b.residuals is None
