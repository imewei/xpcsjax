"""Tests for the argparse-free data service (xpcsjax/service/data.py)."""

from types import SimpleNamespace

import numpy as np

import xpcsjax.service.data as svc_data


def _cm(config: dict) -> SimpleNamespace:
    return SimpleNamespace(config=config, get_config=lambda: config)


def test_load_dataset_subsets_by_phi(monkeypatch):
    base = {
        "c2_exp": np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4),
        "phi_angles_list": np.array([0.0, 45.0, 90.0]),
    }
    monkeypatch.setattr(
        svc_data, "load_xpcs_data", lambda config_dict: {k: v.copy() for k, v in base.items()}
    )
    # Identity filter so the test isolates the --phi subsetting behavior.
    monkeypatch.setattr(svc_data, "apply_angle_filtering_for_optimization", lambda data, cm: data)

    out = svc_data.load_dataset(
        _cm({"analysis_mode": "laminar_flow", "data_type": "aps_old"}), phi_subset=[45.0]
    )
    assert out["c2_exp"].shape[0] == 1
    assert list(out["phi_angles_list"]) == [45.0]


def test_load_dataset_no_subset_keeps_all(monkeypatch):
    base = {
        "c2_exp": np.zeros((3, 4, 4)),
        "phi_angles_list": np.array([0.0, 45.0, 90.0]),
    }
    monkeypatch.setattr(
        svc_data, "load_xpcs_data", lambda config_dict: {k: v.copy() for k, v in base.items()}
    )
    monkeypatch.setattr(svc_data, "apply_angle_filtering_for_optimization", lambda data, cm: data)

    out = svc_data.load_dataset(_cm({"analysis_mode": "laminar_flow"}), phi_subset=None)
    assert out["c2_exp"].shape[0] == 3


def test_load_dataset_raises_when_no_correlation_matrix(monkeypatch):
    monkeypatch.setattr(
        svc_data, "load_xpcs_data", lambda config_dict: {"phi_angles_list": np.array([0.0])}
    )
    monkeypatch.setattr(svc_data, "apply_angle_filtering_for_optimization", lambda data, cm: data)
    import pytest

    with pytest.raises(ValueError, match="no correlation matrix"):
        svc_data.load_dataset(_cm({"analysis_mode": "laminar_flow"}))


def test_subset_data_by_phi_tolerates_empty_data_phi():
    """An empty phi array must warn-and-skip, not raise from ``np.argmin``.

    Regression: ``load_dataset``'s caller-side guard only checks
    ``data_phi_arr is not None``, so a dataset carrying a present-but-empty phi
    key reached ``np.argmin`` on an empty sequence and raised ValueError instead
    of the documented "no match -> fit all angles" behavior.
    """
    data = {
        "c2_exp": np.zeros((0, 4, 4)),
        "phi_angles_list": np.array([]),
    }
    svc_data._subset_data_by_phi(data, np.array([]), [45.0])
    # No slicing performed: the arrays are left exactly as they came in.
    assert data["c2_exp"].shape == (0, 4, 4)
    assert data["phi_angles_list"].size == 0


def test_load_dataset_empty_phi_does_not_crash(monkeypatch):
    """End-to-end path the caller-side ``is not None`` guard failed to cover."""
    monkeypatch.setattr(
        svc_data,
        "load_xpcs_data",
        lambda config_dict: {"c2_exp": np.zeros((0, 4, 4)), "phi_angles_list": np.array([])},
    )
    monkeypatch.setattr(svc_data, "apply_angle_filtering_for_optimization", lambda data, cm: data)

    out = svc_data.load_dataset(_cm({"analysis_mode": "laminar_flow"}), phi_subset=[45.0])
    assert out["phi_angles_list"].size == 0


def test_resolve_phi_angles_cli_wins_and_normalizes():
    cm = _cm({"scattering": {"phi_angles": [10.0, 200.0]}})
    assert svc_data.resolve_phi_angles(cm, cli_phi=[5.0]) == [5.0]
    # No CLI source -> config scattering.phi_angles, normalized to [-180, 180] (200 -> -160).
    assert svc_data.resolve_phi_angles(cm) == [10.0, -160.0]


def test_resolve_phi_angles_parses_comma_string():
    cm = _cm({})
    assert svc_data.resolve_phi_angles(cm, phi_angles_str="0, 45, 90") == [0.0, 45.0, 90.0]


def test_resolve_phi_angles_none_when_no_source():
    assert svc_data.resolve_phi_angles(_cm({})) is None
