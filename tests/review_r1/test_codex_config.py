"""Red tests for Codex review round 1, shard config (claims config-B1..B11).

Kept: CONFIRMED claims only (B1, B3, B4, B6, B7, B8, B10); these fail until fixed."""

from pathlib import Path

import pytest

from xpcsjax.config import ConfigManager, ParameterSpace
from xpcsjax.config.heterodyne_parameter_manager import HeterodyneParameterManager

TEMPLATES = Path(__file__).resolve().parents[2] / "xpcsjax" / "config" / "templates"


def test_config_b1():
    cfg = ConfigManager(
        config_override={
            "experimental_data": {
                "file_path": "/data/chosen.h5",
                "data_folder_path": "/data",
                "data_file_name": "sample.hdf",
            }
        }
    )
    exp = cfg.config["experimental_data"]
    assert exp["file_path"] == "/data/chosen.h5"
    assert Path(exp["data_folder_path"]) / exp["data_file_name"] == Path(exp["file_path"])


@pytest.mark.parametrize(
    "initial, typo",
    [
        ({"active_parameters": ["D0_ref", "D0_sampel"]}, "D0_sampel"),
        ({"fixed_parameters": {"v_offest": 0}}, "v_offest"),
    ],
)
def test_config_b3(initial, typo):
    with pytest.raises(ValueError, match=typo):
        HeterodyneParameterManager.from_config({"initial_parameters": initial})


@pytest.mark.parametrize(
    "cfg",
    [
        {"parameters": {"velocity": {"v0": {"value": 500, "vary": "false"}}}},
        {"parameter_space": {"bounds": [{"name": "v0", "vary": "false"}]}},
    ],
)
def test_config_b4(cfg):
    try:
        pm = HeterodyneParameterManager.from_config(cfg)
    except ValueError:
        return
    assert "v0" not in pm.varying_names


@pytest.mark.parametrize("grouped", [None, {"velocity": None}])
def test_config_b6(grouped):
    pm = HeterodyneParameterManager.from_config(
        {
            "parameters": grouped,
            "initial_parameters": {"parameter_names": ["v0"], "values": [42]},
        }
    )
    assert pm.get_parameter_dict()["v0"] == 42


def test_config_b7():
    cfg = {"parameter_space": {"bounds": [{"name": "D0", "min": None, "max": 900}]}}
    assert ParameterSpace.from_config(cfg).get_bounds("D0") == (100.0, 900.0)


def test_config_b8():
    cfg = ConfigManager(str(TEMPLATES / "xpcsjax_laminar_flow.yaml"))
    assert "gamma_dot_t0" in cfg.get_active_parameters()
    cfg.config_file = str(TEMPLATES / "xpcsjax_static_isotropic.yaml")
    cfg.load_config()
    assert cfg.get_active_parameters() == ["D0", "alpha", "D_offset"]


def test_config_b10():
    pm = HeterodyneParameterManager()
    before = pm.get_full_values()[0]
    with pytest.raises(ValueError, match="unknown"):
        pm.update_values({"D0_ref": 5.0, "unknown": 0.0})
    assert pm.get_parameter_dict()["D0_ref"] == pm.get_full_values()[0] == before
