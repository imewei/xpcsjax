"""Config-driven phi_filtering must subset the data arrays.

Regression: the HDF5 loader only honors a ``data_filtering`` block, not the
top-level ``phi_filtering`` block used by the templates. Without explicit
wiring in ``load_and_validate_data`` the filter selected nothing and the fit
and plots ran on all angles. ``load_and_validate_data`` now mirrors upstream
homodyne by calling ``apply_angle_filtering_for_optimization`` so both the
optimizer and the experimental-data plots see only the selected angles.
"""

from __future__ import annotations

import argparse

import numpy as np

from xpcsjax.cli import data_pipeline
from xpcsjax.config import ConfigManager


def _write_config(tmp_path) -> str:
    """Minimal two_component config with phi_filtering enabled."""
    cfg = tmp_path / "two_component.yaml"
    cfg.write_text(
        """
analysis_mode: "two_component"
analyzer_parameters:
  dt: 1.0
  start_frame: 1
  end_frame: 10
  scattering:
    wavevector_q: 0.01
experimental_data:
  data_folder_path: "/tmp"
  data_file_name: "dummy.hdf"
  phi_angles_path: "/tmp"
  phi_angles_file: "phi.txt"
phi_filtering:
  enabled: true
  target_ranges:
    - min_angle: -10.0
      max_angle: 10.0
    - min_angle: 85.0
      max_angle: 95.0
"""
    )
    return str(cfg)


def _fake_loader_data() -> dict:
    """23-angle synthetic dataset mirroring the C044 azimuthal sweep."""
    phi = np.array(
        [
            -25.5,
            -16.4,
            -5.8,
            4.9,
            15.5,
            26.1,
            36.8,
            47.0,
            58.6,
            68.7,
            79.4,
            90.0,
            100.6,
            111.3,
            121.6,
            132.9,
            143.2,
            153.9,
            164.5,
            175.1,
            185.8,
            196.4,
            205.5,
        ]
    )
    return {
        "phi_angles_list": phi,
        "c2_exp": np.ones((len(phi), 4, 4)),
        "t1": np.arange(4, dtype=float),
        "t2": np.arange(4, dtype=float),
        "wavevector_q_list": np.full(len(phi), 0.0054),
    }


def test_load_and_validate_data_subsets_to_filtered_angles(tmp_path, monkeypatch):
    cfg = ConfigManager(_write_config(tmp_path))
    monkeypatch.setattr(data_pipeline, "load_xpcs_data", lambda **_kw: _fake_loader_data())

    args = argparse.Namespace(phi=None, phi_angles=None)
    out = data_pipeline.load_and_validate_data(args, cfg)

    selected = sorted(round(float(a), 1) for a in np.asarray(out["phi_angles_list"]))
    # [-10, 10] selects -5.8 and 4.9; [85, 95] selects 90.0.
    assert selected == [-5.8, 4.9, 90.0]
    assert out["c2_exp"].shape[0] == 3


def test_phi_filtering_disabled_keeps_all_angles(tmp_path, monkeypatch):
    cfg = ConfigManager(_write_config(tmp_path))
    cfg.config["phi_filtering"]["enabled"] = False
    monkeypatch.setattr(data_pipeline, "load_xpcs_data", lambda **_kw: _fake_loader_data())

    args = argparse.Namespace(phi=None, phi_angles=None)
    out = data_pipeline.load_and_validate_data(args, cfg)

    assert np.asarray(out["phi_angles_list"]).shape[0] == 23
    assert out["c2_exp"].shape[0] == 23
