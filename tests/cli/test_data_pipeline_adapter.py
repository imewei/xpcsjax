"""The CLI data_pipeline adapters must map argparse attrs to service kwargs."""

from types import SimpleNamespace

import xpcsjax.cli.data_pipeline as dp


def test_load_and_validate_data_forwards_phi_as_phi_subset(monkeypatch):
    captured = {}

    def _fake_load_dataset(cm, *, phi_subset=None):
        captured["phi_subset"] = phi_subset
        return {}

    monkeypatch.setattr(dp, "load_dataset", _fake_load_dataset)
    dp.load_and_validate_data(SimpleNamespace(phi=[45.0]), object())
    assert captured["phi_subset"] == [45.0]


def test_resolve_phi_angles_forwards_cli_phi_and_phi_angles_str(monkeypatch):
    captured = {}

    def _fake_resolve(cm, *, cli_phi=None, phi_angles_str=None):
        captured.update(cli_phi=cli_phi, phi_angles_str=phi_angles_str)
        return None

    monkeypatch.setattr(dp, "_service_resolve_phi_angles", _fake_resolve)
    dp.resolve_phi_angles(SimpleNamespace(phi=[1.0], phi_angles="2,3"), object())
    assert captured == {"cli_phi": [1.0], "phi_angles_str": "2,3"}
