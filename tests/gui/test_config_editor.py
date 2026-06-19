import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.views.config_editor import ConfigEditor  # noqa: E402


def test_set_mode_populates_form(qtbot):
    w = ConfigEditor()
    qtbot.addWidget(w)
    w.set_mode("static_isotropic")
    cfg = w.current_config()
    assert cfg["analysis_mode"] == "static_isotropic"


def test_validate_flags_bad_value(qtbot):
    w = ConfigEditor()
    qtbot.addWidget(w)
    w.set_mode("static_isotropic")
    w.set_parameter("D0", -5.0)  # out of bounds
    rep = w.validate()
    assert not rep.ok


def test_raw_yaml_round_trips(qtbot):
    w = ConfigEditor()
    qtbot.addWidget(w)
    w.set_mode("laminar_flow")
    w.toggle_raw(True)
    text = w.raw_text()
    assert "analysis_mode" in text
    w.toggle_raw(False)  # parse back without error
    assert w.current_config()["analysis_mode"] == "laminar_flow"
