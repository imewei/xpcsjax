"""Regression tests for fish-shell XLA activation under conda.

Adversarial-review finding (medium): the fish activation block hardcoded
``$VIRTUAL_ENV/etc/xpcsjax/xla_config.fish``. In a conda fish shell
``$VIRTUAL_ENV`` is normally unset, so the installed hook sourced the wrong
path and XLA settings silently never applied — while post-install reported
success. The block must resolve ``$VIRTUAL_ENV`` with a ``$CONDA_PREFIX``
fallback, matching the bash hook and ``get_venv_path()``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from xpcsjax.post_install import (
    XLA_BEGIN_MARKER,
    XLA_END_MARKER,
    _install_xla_fish_activation,
)


def _make_fake_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "env"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate.fish").write_text("# fake activate.fish\n", encoding="utf-8")
    return venv


def _appended_block(activate_fish: Path) -> str:
    content = activate_fish.read_text(encoding="utf-8")
    start = content.index(XLA_BEGIN_MARKER)
    end = content.index(XLA_END_MARKER) + len(XLA_END_MARKER)
    return content[start:end]


def test_fish_activation_resolves_conda_prefix(tmp_path: Path) -> None:
    venv = _make_fake_venv(tmp_path)
    assert _install_xla_fish_activation(venv, verbose=False) is True

    block = _appended_block(venv / "bin" / "activate.fish")

    # Both environment markers must participate in resolution.
    assert "$VIRTUAL_ENV" in block
    assert "$CONDA_PREFIX" in block
    # The sourced config is resolved relative to whichever prefix won, not a
    # hardcoded $VIRTUAL_ENV path.
    assert "etc/xpcsjax/xla_config.fish" in block
    assert "$VIRTUAL_ENV/etc/xpcsjax/xla_config.fish" not in block


def test_fish_activation_is_idempotent(tmp_path: Path) -> None:
    venv = _make_fake_venv(tmp_path)
    activate = venv / "bin" / "activate.fish"

    assert _install_xla_fish_activation(venv, verbose=False) is True
    assert _install_xla_fish_activation(venv, verbose=False) is True

    content = activate.read_text(encoding="utf-8")
    assert content.count(XLA_BEGIN_MARKER) == 1


def test_fish_activation_missing_script_returns_false(tmp_path: Path) -> None:
    venv = tmp_path / "env"
    (venv / "bin").mkdir(parents=True)  # no activate.fish
    assert _install_xla_fish_activation(venv, verbose=False) is False


def test_generated_fish_block_parses() -> None:
    """If fish is installed, the generated activate.fish must parse cleanly.

    Guards against fish-syntax regressions (e.g. accidentally emitting bash
    ``${VAR:-default}`` expansion, which fish does not understand).
    """
    fish = shutil.which("fish")
    if fish is None:
        import pytest

        pytest.skip("fish shell not installed")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        venv = _make_fake_venv(Path(td))
        _install_xla_fish_activation(venv, verbose=False)
        activate = venv / "bin" / "activate.fish"
        # `fish -n` parses without executing.
        proc = subprocess.run(
            [fish, "-n", str(activate)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
