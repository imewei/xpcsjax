import argparse

import pytest


@pytest.mark.parametrize(
    "modpath",
    [
        "xpcsjax.cli.args_parser",
        "xpcsjax.cli.config_generator",
        "xpcsjax.cli.xla_config",
        "xpcsjax.post_install",
        "xpcsjax.uninstall_scripts",
        "xpcsjax.runtime.utils.system_validator",
    ],
)
def test_build_parser_returns_parser(modpath):
    mod = __import__(modpath, fromlist=["build_parser"])
    parser = mod.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    # Has at least one real option beyond -h
    opts = {s for a in parser._actions for s in a.option_strings}
    assert opts - {"-h", "--help"}


def test_xla_config_threads_roundtrip():
    from xpcsjax.cli import xla_config

    ns = xla_config.build_parser().parse_args(["--threads", "4"])
    assert ns.threads == 4


def test_post_install_shell_choices_preserved():
    from xpcsjax import post_install

    shell_action = next(
        a for a in post_install.build_parser()._actions if "--shell" in a.option_strings
    )
    assert shell_action.choices == ["bash", "zsh", "fish"]


def test_gui_build_parser_import_light():
    # Run in a CLEAN subprocess: other tests in the same pytest process may have
    # already imported PySide6 (and popping "PySide6" does not clear its
    # submodules), so an in-process sys.modules check is flaky. A fresh
    # interpreter is the only reliable assertion.
    import subprocess
    import sys

    code = (
        "import argparse, sys\n"
        "from xpcsjax.gui import app\n"
        "assert isinstance(app.build_parser(), argparse.ArgumentParser)\n"
        "assert 'PySide6' not in sys.modules, sorted(m for m in sys.modules if 'PySide6' in m)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
