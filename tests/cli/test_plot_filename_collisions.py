"""Regression: per-angle plot filenames must not collide.

Bug history: the three plot families named their per-angle PNGs from
``int(round(phi))`` alone, so any two angles under 1 degree apart (normal for
fine azimuthal binning) produced the same path and one plot silently
overwrote the other. ``experimental.py`` was worse: with an empty
``phi_angles_list`` every frame fell back to ``phi=0.0`` and collapsed onto a
single file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import xpcsjax.viz as viz
from xpcsjax.cli.plot_families import experimental, postfit, simulated
from xpcsjax.config import ConfigManager


def _write_config(tmp_path) -> str:
    cfg = tmp_path / "static.yaml"
    cfg.write_text(
        """
analysis_mode: "static_isotropic"
analyzer_parameters:
  dt: 0.5
  start_frame: 1
  end_frame: 6
  scattering:
    wavevector_q: 0.0054
experimental_data:
  data_folder_path: "/tmp"
  data_file_name: "dummy.hdf"
"""
    )
    return str(cfg)


def test_simulated_close_angles_get_distinct_files(tmp_path, monkeypatch):
    cm = ConfigManager(_write_config(tmp_path))
    paths: list[Path] = []

    monkeypatch.setattr(simulated, "_evaluate_model_c2", lambda *a, **k: np.zeros((6, 6)))
    monkeypatch.setattr(viz, "plot_simulated_data", lambda *a, **k: paths.append(k["save_path"]))

    simulated._plot_simulated_from_config(
        cm,
        contrast=0.5,
        offset=1.0,
        phi_angles_str="12.3,12.6",
        plots_dir=Path(tmp_path),
        data=None,
    )

    assert len(paths) == 2
    assert len(set(paths)) == 2


def test_experimental_missing_phi_list_gets_distinct_files(tmp_path, monkeypatch):
    paths: list[Path] = []
    monkeypatch.setattr(viz, "plot_simulated_data", lambda *a, **k: paths.append(k["save_path"]))

    experimental._plot_experimental_data({"c2_exp": np.zeros((3, 4, 4))}, Path(tmp_path))

    assert len(paths) == 3
    assert len(set(paths)) == 3


def test_postfit_close_angles_get_distinct_files(tmp_path, monkeypatch):
    from xpcsjax.viz import nlsq_plots

    fits: list[Path] = []
    residuals: list[Path] = []

    class _CM:
        config: dict = {}

        def get_model(self):
            return object()

    class _Result:
        reduced_chi_squared = 1.0

    monkeypatch.setattr(nlsq_plots, "_evaluate_c2_per_angle", lambda *a, **k: np.zeros((4, 4)))
    monkeypatch.setattr(viz, "plot_nlsq_fit", lambda *a, **k: fits.append(k["save_path"]))
    monkeypatch.setattr(viz, "plot_residual_map", lambda *a, **k: residuals.append(k["save_path"]))

    postfit._save_fit_comparison_only(
        _CM(),
        {"c2_exp": np.zeros((2, 4, 4)), "phi_angles_list": [12.3, 12.6]},
        _Result(),
        Path(tmp_path),
    )

    assert len(fits) == 2
    assert len(set(fits)) == 2
    assert len(set(residuals)) == 2
