"""Regression: the standalone ``--plot-simulated-data`` path must evaluate the
heterodyne model on the configured *elapsed-time* grid, not on frame indices.

Bug history: ``_plot_simulated_from_config`` built a bare ``np.arange(64)``
frame-index axis (or fed raw frame-index data arrays) into the time-dependent
two-component model. Because the cross term is ``cos(q·cos φ·∫v(t')dt')``,
integrating over the wrong grid collapsed the fringe structure and produced a
qualitatively wrong C2 surface (axis 0..63 instead of 0..100 s, fanning
fringes lost). The grid must mirror ``heterodyne.HeterodyneModel.from_config``:
``t = arange(n_times) * dt + dt`` with ``n_times = end_frame - start_frame + 1``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import xpcsjax.viz as viz
from xpcsjax.cli import plot_dispatch
from xpcsjax.config import ConfigManager


def _write_config(tmp_path) -> str:
    cfg = tmp_path / "two_component.yaml"
    cfg.write_text(
        """
analysis_mode: "two_component"
analyzer_parameters:
  dt: 0.5
  start_frame: 1
  end_frame: 10
  scattering:
    wavevector_q: 0.0054
experimental_data:
  data_folder_path: "/tmp"
  data_file_name: "dummy.hdf"
"""
    )
    return str(cfg)


def test_simulated_grid_uses_elapsed_time(tmp_path, monkeypatch):
    cm = ConfigManager(_write_config(tmp_path))

    captured: dict[str, np.ndarray] = {}

    def _capture(model, params, phi, t1, t2, **kw):  # noqa: ANN001
        captured["t1"] = np.asarray(t1)
        captured["dt"] = kw["dt"]
        n = len(np.asarray(t1))
        return np.zeros((n, n))

    monkeypatch.setattr(plot_dispatch, "_evaluate_model_c2", _capture)
    monkeypatch.setattr(viz, "plot_simulated_data", lambda *a, **k: None)

    plot_dispatch._plot_simulated_from_config(
        cm, contrast=0.5, offset=1.0, phi_angles_str="0",
        plots_dir=Path(tmp_path), data=None,
    )

    # n_times = end_frame - start_frame + 1 = 10; t_start = dt = 0.5.
    expected = np.arange(10, dtype=np.float64) * 0.5 + 0.5
    np.testing.assert_allclose(captured["t1"], expected)
    assert captured["dt"] == 0.5
    # Guard against the regression: a bare arange would start at 0.0.
    assert captured["t1"][0] > 0.0
