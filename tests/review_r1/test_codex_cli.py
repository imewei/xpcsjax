"""Red tests for the Codex round-1 review of cli/runtime/device.

Confirmed claims cli-B1..B8 and B10; B9 was judged DESIGN and has no test here.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}


def test_cli_b1(tmp_path):
    from xpcsjax.cli.config_template import generate_config
    from xpcsjax.config import ConfigManager
    from xpcsjax.data.xpcs_loader import XPCSDataLoader

    requested = tmp_path / "experiment.h5"
    path = generate_config("static_anisotropic", tmp_path / "cfg.yaml", data_path=str(requested))
    cfg = ConfigManager(str(path)).config
    # The loader must open the requested file, not the template's ./data/sample.hdf...
    exp = XPCSDataLoader(config_dict=cfg).exp_config
    opened = Path(exp["data_folder_path"]) / exp["data_file_name"]
    assert opened == requested
    # ...and the manager must honour the template's "file_path wins" promise.
    assert cfg["experimental_data"]["file_path"] == str(requested)


def _stub_fit(monkeypatch, c):
    monkeypatch.setattr(c, "load_and_validate_data", lambda *a: {})
    monkeypatch.setattr(c, "resolve_phi_angles", lambda *a: None)
    monkeypatch.setattr(c, "run_nlsq", lambda *a: SimpleNamespace(success=True))
    monkeypatch.setattr(c, "save_results", lambda *a, **k: None)


def test_cli_b2(tmp_path, monkeypatch):
    from xpcsjax.cli import commands as c

    old = tmp_path / "nlsq_result.npz"
    np.savez(old, parameters=[1.0])
    os.utime(old, (1_000_000, 1_000_000))  # run A, long ago
    before = old.read_bytes()

    def plot(*args, **kwargs):
        sidecar = tmp_path / "plots/simulated_data/c2_fitted_data.npz"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sidecar, c2_exp=[[[2.0]]])
        return 0

    _stub_fit(monkeypatch, c)
    monkeypatch.setattr(c, "dispatch_plots", plot)
    args = SimpleNamespace(output=tmp_path, output_format="json", plot=True)
    assert c._dispatch_fit(args, None) == 0
    assert old.read_bytes() == before


def test_cli_b3():
    code = (
        "import resource, sys; resource.setrlimit(resource.RLIMIT_CORE, (0, 0));"
        "from xpcsjax.cli.main import _bootstrap_xla_env;"
        "_bootstrap_xla_env(sys.argv[1:]); import jax; assert jax.devices('cpu')"
    )

    def run(*argv):
        return subprocess.run(
            [sys.executable, "-B", "-c", code, *argv],
            env=_clean_env(),
            capture_output=True,
            text=True,
            timeout=300,
        )

    assert run().returncode == 0  # control
    proc = run("--threads", "2")
    assert proc.returncode == 0, proc.stderr[-500:]


def _sim(tmp_path, monkeypatch, override):
    import xpcsjax.viz as viz
    from xpcsjax.cli.plot_families import simulated as s
    from xpcsjax.config import ConfigManager

    cm = ConfigManager(config_override=override)
    seen = []

    def evaluate(model, params, *args, **kwargs):
        seen.append((np.asarray(params).tolist(), kwargs["q"]))
        return np.ones((3, 3))

    monkeypatch.setattr(s, "_evaluate_model_c2", evaluate)
    monkeypatch.setattr(viz, "plot_simulated_data", lambda *a, **k: None)
    s._plot_simulated_from_config(cm, 0.3, 1.0, "0", tmp_path, None)
    return seen


def test_cli_b4(tmp_path, monkeypatch):
    seen = _sim(
        tmp_path,
        monkeypatch,
        {
            "analysis_mode": "two_component",
            "temporal": {"dt": 0.1, "time_length": 3},
            "scattering": {"wavevector_q": 0.0237},
        },
    )
    assert seen, "model was never evaluated"
    assert {q for _, q in seen} == {0.0237}


def test_cli_b5(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        seen = _sim(
            tmp_path,
            monkeypatch,
            {
                "analysis_mode": "static_isotropic",
                "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 3},
                "initial_parameters": {
                    "parameter_names": ["D0", "alpha", "D_offset"],
                    "values": [1000.0, 1.0, 0.0],
                    "fixed_parameters": {"D_offset": 0.0},
                },
            },
        )
    assert seen, f"model was never evaluated: {caplog.text[-400:]}"
    assert {tuple(p) for p, _ in seen} == {(1000.0, 1.0, 0.0)}


def test_cli_b6(tmp_path):
    from xpcsjax.cli.config_template import validate_config_file

    for text in ("", "42\n"):
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        assert not validate_config_file(path).ok, repr(text)


def test_cli_b7(tmp_path, monkeypatch):
    from xpcsjax.cli import commands as c

    (tmp_path / "plots").touch()
    _stub_fit(monkeypatch, c)
    args = SimpleNamespace(output=tmp_path, output_format="both", plot=True)
    assert c._dispatch_fit(args, None) == 0


def test_cli_b8(tmp_path):
    import yaml

    from xpcsjax.cli.config_template import generate_config

    path = "/data/run\n2026.h5"
    out = generate_config("static_isotropic", tmp_path / "cfg.yaml", data_path=path)
    assert yaml.safe_load(out.read_text())["experimental_data"]["file_path"] == path


# cli-B10 reclassified DESIGN by the lead (fix changes the public xpcsjax.cli __all__); test removed.
