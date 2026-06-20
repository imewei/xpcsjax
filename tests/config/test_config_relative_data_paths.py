"""Relative data paths in a config resolve against the config file's directory.

Regression for the GUI ``FileNotFoundError`` on configs that use paths relative
to themselves (e.g. ``data_folder_path: "./"``). The CLI happened to work only
because users ``cd`` into the data directory first; the GUI worker subprocess
runs from a different working directory, so ``./`` pointed at the wrong place.

The fix anchors relative ``experimental_data`` path keys to
``Path(config_file).parent`` at load time (the one place the config-file
location is still known). Absolute paths and unresolved ``${VARS}`` are left
untouched, so the rtol=1e-10 parity baselines (absolute / ``${XPCSJAX_DATA_ROOT}``
fixtures) are unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xpcsjax.config.manager import ConfigManager

yaml = pytest.importorskip("yaml")

_PHYSICS = {
    "analysis_mode": "laminar_flow",
}


def _write_config(tmp_path: Path, exp: dict) -> Path:
    cfg = dict(_PHYSICS)
    cfg["experimental_data"] = exp
    p = tmp_path / "xpcsjax_config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_relative_data_folder_anchored_to_config_dir(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "data_folder_path": "./",
            "data_file_name": "data.hdf",
            "file_path": "./data.hdf",
            "phi_angles_path": "./",
            "cache_directory": "./",
        },
    )
    cm = ConfigManager(config_file=str(cfg_path))
    exp = cm.config["experimental_data"]
    base = str(tmp_path.resolve())

    # data_folder_path "./" -> the config's own directory (absolute).
    assert os.path.isabs(exp["data_folder_path"])
    assert os.path.normpath(exp["data_folder_path"]) == base
    # file_path "./data.hdf" -> <config dir>/data.hdf
    assert os.path.normpath(exp["file_path"]) == os.path.join(base, "data.hdf")
    assert os.path.normpath(exp["phi_angles_path"]) == base
    # basenames are left alone (they are joined to their folder downstream).
    assert exp["data_file_name"] == "data.hdf"


def test_relative_subdir_is_joined(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, {"data_folder_path": "raw/sub"})
    cm = ConfigManager(config_file=str(cfg_path))
    got = os.path.normpath(cm.config["experimental_data"]["data_folder_path"])
    assert got == os.path.join(str(tmp_path.resolve()), "raw", "sub")


def test_absolute_path_is_left_unchanged(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, {"data_folder_path": "/abs/data", "file_path": "/abs/x.hdf"})
    cm = ConfigManager(config_file=str(cfg_path))
    exp = cm.config["experimental_data"]
    assert exp["data_folder_path"] == "/abs/data"
    assert exp["file_path"] == "/abs/x.hdf"


def test_env_var_path_expands_to_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPCSJAX_TEST_ROOT", "/srv/scratch")
    cfg_path = _write_config(tmp_path, {"data_folder_path": "${XPCSJAX_TEST_ROOT}/C020"})
    cm = ConfigManager(config_file=str(cfg_path))
    got = cm.config["experimental_data"]["data_folder_path"]
    assert os.path.normpath(got) == os.path.join("/srv", "scratch", "C020")


def test_unset_env_var_is_not_mis_anchored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPCSJAX_DEFINITELY_UNSET", raising=False)
    cfg_path = _write_config(tmp_path, {"data_folder_path": "${XPCSJAX_DEFINITELY_UNSET}/C020"})
    cm = ConfigManager(config_file=str(cfg_path))
    # An unresolved ${VAR} must NOT be joined to the config dir (that would
    # silently point at the wrong tree); leave it for an honest downstream error.
    got = cm.config["experimental_data"]["data_folder_path"]
    assert "${XPCSJAX_DEFINITELY_UNSET}" in got
    assert str(tmp_path.resolve()) not in got


def test_override_path_is_not_resolved() -> None:
    # config_override never reads a file, so there is no config dir to anchor to;
    # relative paths must pass through untouched.
    cm = ConfigManager(
        config_override={
            "analysis_mode": "laminar_flow",
            "experimental_data": {"data_folder_path": "./"},
        }
    )
    assert cm.config["experimental_data"]["data_folder_path"] == "./"
