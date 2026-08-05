"""The CLI data_pipeline adapters must map argparse attrs to service kwargs."""

import logging
from types import SimpleNamespace

import xpcsjax.cli.commands as cmds
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


def test_dispatch_fit_is_non_fatal_on_malformed_phi_angles(monkeypatch, caplog):
    """A malformed --phi-angles must not abort the whole fit run — it is only
    ever used for one informational log line here, and validate_args() already
    treats the same malformed input as non-fatal (2026-08-05 pr-review-toolkit
    pr-test-analyzer finding, PR #36). This is distinct from resolve_phi_angles
    itself, which is expected to keep raising ValueError on bad input — see
    test_resolve_phi_angles_forwards_cli_phi_and_phi_angles_str above.
    """
    monkeypatch.setattr(cmds, "load_and_validate_data", lambda args, cm: {})
    monkeypatch.setattr(
        cmds,
        "resolve_phi_angles",
        lambda args, cm: (_ for _ in ()).throw(ValueError("bad phi-angles")),
    )
    monkeypatch.setattr(cmds, "run_nlsq", lambda args, cm, data: SimpleNamespace(success=True))
    monkeypatch.setattr(cmds, "_resolve_output_dir", lambda args, cm: None)

    args = SimpleNamespace(phi_angles="1,not_a_number,3", plot=False, save_plots=False)
    with caplog.at_level(logging.WARNING, logger="xpcsjax.cli.commands"):
        result_code = cmds._dispatch_fit(args, object())

    assert result_code == 0, "a resolvable NLSQ run must not abort over bad --phi-angles"
    assert any("Could not resolve phi angles" in r.message for r in caplog.records)
