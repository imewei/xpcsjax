"""Tests for xpcsjax.uninstall_scripts.

Covers venv-path resolution, cleanup-target discovery, completion/XLA file
removal (real + dry-run), the pure ``_remove_xpcsjax_blocks`` state machine
(explicit markers + legacy depth-tracked blocks, bash ``fi`` + fish ``end``),
activation-script scrubbing, and the ``main`` CLI dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from xpcsjax import uninstall_scripts as us


@pytest.fixture
def venv_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Sandbox HOME + VIRTUAL_ENV to tmp and return the fake venv path."""
    home = tmp_path / "home"
    home.mkdir()
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # Neutralize the sys.prefix branch so get_venv_path uses VIRTUAL_ENV.
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    return venv


# ---------------------------------------------------------------------------
# get_venv_path
# ---------------------------------------------------------------------------


def test_get_venv_path_via_sys_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix + "_x")
    assert us.get_venv_path() == Path(sys.prefix)


def test_get_venv_path_virtual_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
    assert us.get_venv_path() == Path("/some/venv")


def test_get_venv_path_conda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/z")
    assert us.get_venv_path() == Path("/opt/conda/envs/z")


def test_get_venv_path_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert us.get_venv_path() is None


# ---------------------------------------------------------------------------
# find_cleanup_targets
# ---------------------------------------------------------------------------


def test_find_cleanup_targets_marks_existing(venv_env: Path) -> None:
    bash_completion = venv_env / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
    bash_completion.parent.mkdir(parents=True)
    bash_completion.write_text("# completion\n", encoding="utf-8")

    targets = us.find_cleanup_targets()
    by_path = {t.path: t for t in targets}
    assert bash_completion in by_path
    assert by_path[bash_completion].exists is True

    # A path that was never created is listed but flagged non-existent.
    zsh = venv_env / "etc" / "zsh" / "xpcsjax-completion.sh"
    assert zsh in by_path
    assert by_path[zsh].exists is False


def test_find_cleanup_targets_detects_legacy_xla_mode(venv_env: Path) -> None:
    legacy = Path.home() / ".xpcsjax_xla_mode"
    legacy.write_text("auto", encoding="utf-8")
    targets = us.find_cleanup_targets()
    descs = [t.description for t in targets if t.exists]
    assert any("legacy" in d.lower() for d in descs)


def test_find_cleanup_targets_flags_empty_parent_dir(venv_env: Path) -> None:
    # Create the XLA etc dir but leave it empty -> flagged as empty directory.
    empty_dir = venv_env / "etc" / "xpcsjax"
    empty_dir.mkdir(parents=True)
    targets = us.find_cleanup_targets()
    assert any(t.description.startswith("Empty directory") and t.path == empty_dir for t in targets)


# ---------------------------------------------------------------------------
# cleanup_completion_files / cleanup_xla_config
# ---------------------------------------------------------------------------


def test_cleanup_completion_files_removes(venv_env: Path) -> None:
    comp = venv_env / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
    comp.parent.mkdir(parents=True)
    comp.write_text("x", encoding="utf-8")
    removed = us.cleanup_completion_files(verbose=True)
    assert comp in removed
    assert not comp.exists()


def test_cleanup_completion_files_dry_run(venv_env: Path) -> None:
    comp = venv_env / "share" / "fish" / "vendor_completions.d" / "xpcsjax.fish"
    comp.parent.mkdir(parents=True)
    comp.write_text("x", encoding="utf-8")
    removed = us.cleanup_completion_files(dry_run=True, verbose=True)
    assert comp in removed
    assert comp.exists()  # dry-run must not delete


def test_cleanup_xla_config_removes_and_prunes_parent(venv_env: Path) -> None:
    xla = venv_env / "etc" / "xpcsjax" / "xla_config.bash"
    xla.parent.mkdir(parents=True)
    xla.write_text("x", encoding="utf-8")
    removed = us.cleanup_xla_config(verbose=True)
    assert xla in removed
    assert not xla.exists()
    # Empty parent dir is pruned.
    assert not xla.parent.exists()


def test_cleanup_xla_config_dry_run(venv_env: Path) -> None:
    xla = venv_env / "etc" / "xpcsjax" / "xla_config.fish"
    xla.parent.mkdir(parents=True)
    xla.write_text("x", encoding="utf-8")
    removed = us.cleanup_xla_config(dry_run=True)
    assert xla in removed
    assert xla.exists()


# ---------------------------------------------------------------------------
# _remove_xpcsjax_blocks (pure state machine)
# ---------------------------------------------------------------------------


def test_remove_blocks_explicit_markers_bash() -> None:
    content = (
        "export A=1\n"
        "# >>> xpcsjax xla_config >>>\n"
        'if [ -f "x" ]; then\n'
        "    source x\n"
        "fi\n"
        "# <<< xpcsjax xla_config <<<\n"
        "export B=2\n"
    )
    out = us._remove_xpcsjax_blocks(content, "fi")
    assert "xpcsjax" not in out
    assert "export A=1" in out
    assert "export B=2" in out


def test_remove_blocks_explicit_markers_fish() -> None:
    content = (
        "set -x A 1\n"
        "# >>> xpcsjax completion >>>\n"
        'if test -f "x"\n'
        "    source x\n"
        "end\n"
        "# <<< xpcsjax completion <<<\n"
        "set -x B 2\n"
    )
    out = us._remove_xpcsjax_blocks(content, "end")
    assert "xpcsjax" not in out
    assert "set -x A 1" in out
    assert "set -x B 2" in out


def test_remove_blocks_legacy_bash_with_nested_if() -> None:
    content = (
        "keep1\n"
        "# xpcsjax XLA configuration\n"
        'if [ -f "a" ]; then\n'
        '    if [ -n "$Z" ]; then\n'
        "        source a\n"
        "    fi\n"
        "fi\n"
        "keep2\n"
    )
    out = us._remove_xpcsjax_blocks(content, "fi")
    assert "source a" not in out
    assert "xpcsjax" not in out
    assert "keep1" in out
    assert "keep2" in out


def test_remove_blocks_legacy_fish_with_nested_if() -> None:
    content = (
        "keep1\n"
        "# xpcsjax XLA configuration (auto-added by xpcsjax-post-install)\n"
        "if test -f a\n"
        '    if test -n "$Z"\n'
        "        source a\n"
        "    end\n"
        "end\n"
        "keep2\n"
    )
    out = us._remove_xpcsjax_blocks(content, "end")
    assert "source a" not in out
    assert "keep1" in out
    assert "keep2" in out


def test_remove_blocks_noop_without_xpcsjax() -> None:
    content = "line1\nline2\n"
    assert us._remove_xpcsjax_blocks(content, "fi") == content


# ---------------------------------------------------------------------------
# cleanup_xla_activation_scripts
# ---------------------------------------------------------------------------


def test_cleanup_activation_scripts_scrubs_block(venv_env: Path) -> None:
    activate = venv_env / "bin" / "activate"
    activate.write_text(
        "export BASE=1\n"
        "# >>> xpcsjax xla_config >>>\n"
        'if [ -f "x" ]; then\n    source x\nfi\n'
        "# <<< xpcsjax xla_config <<<\n",
        encoding="utf-8",
    )
    assert us.cleanup_xla_activation_scripts(verbose=True) is True
    assert "xpcsjax" not in activate.read_text(encoding="utf-8")
    assert "export BASE=1" in activate.read_text(encoding="utf-8")


def test_cleanup_activation_scripts_dry_run(venv_env: Path) -> None:
    activate = venv_env / "bin" / "activate"
    activate.write_text(
        "# >>> xpcsjax xla_config >>>\nx\n# <<< xpcsjax xla_config <<<\n",
        encoding="utf-8",
    )
    assert us.cleanup_xla_activation_scripts(dry_run=True, verbose=True) is True
    assert "xpcsjax" in activate.read_text(encoding="utf-8")  # unchanged on dry-run


def test_cleanup_activation_scripts_skips_clean_files(venv_env: Path) -> None:
    activate = venv_env / "bin" / "activate"
    activate.write_text("export BASE=1\n", encoding="utf-8")
    # No xpcsjax content -> nothing modified.
    assert us.cleanup_xla_activation_scripts(verbose=True) is False


def test_cleanup_activation_scripts_no_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(us, "get_venv_path", lambda: None)
    assert us.cleanup_xla_activation_scripts() is False


# ---------------------------------------------------------------------------
# show_dry_run / interactive_cleanup
# ---------------------------------------------------------------------------


def test_show_dry_run_no_files(venv_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    us.show_dry_run()
    assert "No xpcsjax files found" in capsys.readouterr().out


def test_show_dry_run_lists_files_and_activation(
    venv_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    comp = venv_env / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
    comp.parent.mkdir(parents=True)
    comp.write_text("x", encoding="utf-8")
    activate = venv_env / "bin" / "activate"
    activate.write_text("# >>> xpcsjax xla_config >>>\nx\n", encoding="utf-8")

    us.show_dry_run(verbose=True)
    out = capsys.readouterr().out
    assert "xpcsjax-completion.sh" in out
    assert "would modify" in out.lower()


def test_interactive_cleanup_cancelled(
    venv_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    comp = venv_env / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
    comp.parent.mkdir(parents=True)
    comp.write_text("x", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _p="": "n")
    us.interactive_cleanup()
    assert "cancelled" in capsys.readouterr().out.lower()
    assert comp.exists()  # cancellation leaves files


def test_interactive_cleanup_confirmed(
    venv_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    comp = venv_env / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
    comp.parent.mkdir(parents=True)
    comp.write_text("x", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _p="": "y")
    us.interactive_cleanup()
    assert "Cleanup complete!" in capsys.readouterr().out
    assert not comp.exists()


def test_interactive_cleanup_nothing_to_do(
    venv_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    us.interactive_cleanup()
    assert "No xpcsjax files found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_dry_run(venv_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert us.main(["--dry-run"]) == 0
    assert "Dry run" in capsys.readouterr().out


def test_main_force(venv_env: Path) -> None:
    comp = venv_env / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
    comp.parent.mkdir(parents=True)
    comp.write_text("x", encoding="utf-8")
    assert us.main(["--force", "--verbose"]) == 0
    assert not comp.exists()


def test_main_default_interactive(venv_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(us, "interactive_cleanup", lambda: called.append(True))
    assert us.main([]) == 0
    assert called == [True]
