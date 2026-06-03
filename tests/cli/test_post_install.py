"""Tests for xpcsjax.post_install.

Covers environment/shell detection, venv-path resolution, completion + XLA
script installation into a fake venv, activation-block injection (bash/zsh +
completion), XLA-mode file management with legacy migration, mode validation,
and the ``main`` CLI dispatch (non-interactive + interactive paths).

The fish XLA activation block has its own focused suite in
``test_post_install_fish.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from xpcsjax import post_install as pi

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_venv(tmp_path: Path) -> Path:
    """A venv-shaped directory with empty bash + fish activate scripts."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("# activate\n", encoding="utf-8")
    (venv / "bin" / "activate.fish").write_text("# activate.fish\n", encoding="utf-8")
    return venv


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point HOME at a tmp dir and clear venv markers so config paths are sandboxed."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # Path.home() reads USERPROFILE (then HOMEDRIVE+HOMEPATH) on Windows, not
    # HOME — sandbox those too so the fixture isolates the home dir on every OS.
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOMEDRIVE", home.drive or "")
    monkeypatch.setenv("HOMEPATH", str(home)[len(home.drive) :] if home.drive else str(home))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home


# ---------------------------------------------------------------------------
# Shell + environment detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shell_path", "expected"),
    [
        ("/usr/bin/zsh", "zsh"),
        ("/bin/bash", "bash"),
        ("/usr/local/bin/fish", "fish"),
    ],
)
def test_detect_shell_type_from_env(
    monkeypatch: pytest.MonkeyPatch, shell_path: str, expected: str
) -> None:
    monkeypatch.setenv("SHELL", shell_path)
    assert pi.detect_shell_type() == expected


def test_detect_shell_type_unknown_via_psutil_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHELL", "")  # force the parent-process fallback
    import psutil

    class _Parent:
        def name(self) -> str:
            return "zsh"

    class _Proc:
        def parent(self) -> _Parent:
            return _Parent()

    monkeypatch.setattr(psutil, "Process", lambda: _Proc())
    assert pi.detect_shell_type() == "zsh"


def test_detect_shell_type_unknown_when_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHELL", "/bin/dash")  # not a recognized name
    import psutil

    def _raise() -> object:
        raise OSError("no parent")

    monkeypatch.setattr(psutil, "Process", _raise)
    assert pi.detect_shell_type() == "unknown"


def test_is_virtual_environment_via_base_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix + "_different")
    assert pi.is_virtual_environment() is True


def test_is_virtual_environment_via_conda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/x")
    assert pi.is_virtual_environment() is True


def test_is_virtual_environment_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert pi.is_virtual_environment() is False


def test_is_conda_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
    assert pi.is_conda_environment() is True
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert pi.is_conda_environment() is False


def test_get_venv_path_prefers_virtual_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/a/venv")
    assert pi.get_venv_path() == Path("/a/venv")


def test_get_venv_path_conda_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/y")
    assert pi.get_venv_path() == Path("/opt/conda/envs/y")


def test_get_venv_path_sys_prefix_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert pi.get_venv_path() == Path(sys.prefix)


# ---------------------------------------------------------------------------
# Source-path resolution
# ---------------------------------------------------------------------------


def test_completion_and_xla_source_paths_exist() -> None:
    assert pi.get_completion_source_path().is_file()
    assert pi.get_xla_config_source_path("bash").is_file()
    assert pi.get_xla_config_source_path("fish").is_file()
    # Non-fish shells map to the bash script.
    assert pi.get_xla_config_source_path("zsh") == pi.get_xla_config_source_path("bash")


# ---------------------------------------------------------------------------
# Completion installation
# ---------------------------------------------------------------------------


def test_install_bash_completion(fake_venv: Path) -> None:
    assert pi.install_bash_completion(fake_venv, verbose=True) is True
    dest = fake_venv / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
    assert dest.is_file()


def test_install_bash_completion_missing_source(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pi, "get_completion_source_path", lambda: Path("/nope/x.sh"))
    assert pi.install_bash_completion(fake_venv, verbose=True) is False


def test_install_zsh_completion(fake_venv: Path) -> None:
    assert pi.install_zsh_completion(fake_venv, verbose=True) is True
    zsh_dest = fake_venv / "etc" / "zsh" / "xpcsjax-completion.sh"
    assert zsh_dest.is_file()
    content = zsh_dest.read_text(encoding="utf-8")
    assert "bashcompinit" in content
    # The prerequisite bash completion is installed too.
    assert (fake_venv / "etc" / "bash_completion.d" / "xpcsjax-completion.sh").is_file()


def test_install_fish_completion(fake_venv: Path) -> None:
    assert pi.install_fish_completion(fake_venv, verbose=True) is True
    dest = fake_venv / "share" / "fish" / "vendor_completions.d" / "xpcsjax.fish"
    assert dest.is_file()
    assert "complete -c xpcsjax" in dest.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("shell", "marker_path"),
    [
        ("bash", "etc/bash_completion.d/xpcsjax-completion.sh"),
        ("zsh", "etc/zsh/xpcsjax-completion.sh"),
        ("fish", "share/fish/vendor_completions.d/xpcsjax.fish"),
        ("unknown", "etc/bash_completion.d/xpcsjax-completion.sh"),
    ],
)
def test_install_shell_completion_routes(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch, shell: str, marker_path: str
) -> None:
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: True)
    monkeypatch.setattr(pi, "get_venv_path", lambda: fake_venv)
    assert pi.install_shell_completion(shell, verbose=True) is True
    assert (fake_venv / marker_path).is_file()


def test_install_shell_completion_skips_outside_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: False)
    assert pi.install_shell_completion("bash", verbose=True) is False


# ---------------------------------------------------------------------------
# Completion activation hooks
# ---------------------------------------------------------------------------


def test_completion_bash_activation_injection_and_idempotency(fake_venv: Path) -> None:
    activate = fake_venv / "bin" / "activate"
    assert pi._install_completion_bash_activation(fake_venv, verbose=True) is True
    text = activate.read_text(encoding="utf-8")
    assert pi.COMPLETION_BEGIN_MARKER in text
    # Second call is a no-op (marker already present).
    assert pi._install_completion_bash_activation(fake_venv, verbose=True) is True
    assert activate.read_text(encoding="utf-8").count(pi.COMPLETION_BEGIN_MARKER) == 1


def test_completion_fish_activation_injection(fake_venv: Path) -> None:
    assert pi._install_completion_fish_activation(fake_venv, verbose=True) is True
    text = (fake_venv / "bin" / "activate.fish").read_text(encoding="utf-8")
    assert pi.COMPLETION_BEGIN_MARKER in text
    assert "vendor_completions.d/xpcsjax.fish" in text


def test_completion_activation_missing_script(tmp_path: Path) -> None:
    empty = tmp_path / "venv"
    (empty / "bin").mkdir(parents=True)
    assert pi._install_completion_bash_activation(empty, verbose=True) is False
    assert pi._install_completion_fish_activation(empty, verbose=True) is False


def test_install_completion_activation_routes(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: True)
    monkeypatch.setattr(pi, "get_venv_path", lambda: fake_venv)
    assert pi.install_completion_activation("fish", verbose=True) is True
    assert pi.install_completion_activation("bash", verbose=True) is True


def test_install_completion_activation_skips_outside_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: False)
    assert pi.install_completion_activation("bash", verbose=True) is False


# ---------------------------------------------------------------------------
# XLA config script installation + activation
# ---------------------------------------------------------------------------


def test_install_xla_config_scripts(fake_venv: Path) -> None:
    assert pi.install_xla_config_scripts(fake_venv, verbose=True) is True
    assert (fake_venv / "etc" / "xpcsjax" / "xla_config.bash").is_file()
    assert (fake_venv / "etc" / "xpcsjax" / "xla_config.fish").is_file()
    assert (
        fake_venv / "share" / "fish" / "vendor_conf.d" / "xpcsjax-xla.fish"
    ).is_file()


def test_install_xla_bash_activation_injection_and_idempotency(fake_venv: Path) -> None:
    activate = fake_venv / "bin" / "activate"
    assert pi._install_xla_bash_activation(fake_venv, verbose=True) is True
    text = activate.read_text(encoding="utf-8")
    assert pi.XLA_BEGIN_MARKER in text
    # Resolves the prefix relocation-safely.
    assert "${VIRTUAL_ENV:-${CONDA_PREFIX}}/etc/xpcsjax/xla_config.bash" in text
    assert pi._install_xla_bash_activation(fake_venv, verbose=True) is True
    assert activate.read_text(encoding="utf-8").count(pi.XLA_BEGIN_MARKER) == 1


def test_install_xla_bash_activation_missing_script(tmp_path: Path) -> None:
    empty = tmp_path / "venv"
    (empty / "bin").mkdir(parents=True)
    assert pi._install_xla_bash_activation(empty, verbose=True) is False


def test_install_xla_activation_routes_bash(
    fake_venv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: True)
    monkeypatch.setattr(pi, "get_venv_path", lambda: fake_venv)
    assert pi.install_xla_activation("bash", verbose=True) is True
    assert pi.XLA_BEGIN_MARKER in (fake_venv / "bin" / "activate").read_text("utf-8")


def test_install_xla_activation_skips_outside_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: False)
    assert pi.install_xla_activation("bash", verbose=True) is False


# ---------------------------------------------------------------------------
# XLA mode path + configuration
# ---------------------------------------------------------------------------


def test_get_xla_mode_path_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    assert pi.get_xla_mode_path() == tmp_path / "venv" / "etc" / "xpcsjax" / "xla_mode"


def test_get_xla_mode_path_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert pi.get_xla_mode_path() == tmp_path / "cfg" / "xpcsjax" / "xla_mode"


def test_get_xla_mode_path_home_fallback(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    assert pi.get_xla_mode_path() == isolated_env / ".config" / "xpcsjax" / "xla_mode"


def test_migrate_legacy_xla_mode(isolated_env: Path, tmp_path: Path) -> None:
    legacy = isolated_env / ".xpcsjax_xla_mode"
    legacy.write_text("nlsq", encoding="utf-8")
    new_path = tmp_path / "cfg" / "xla_mode"
    pi._migrate_legacy_xla_mode(new_path)
    assert new_path.read_text(encoding="utf-8") == "nlsq"
    assert not legacy.exists()


def test_migrate_legacy_noop_when_new_exists(
    isolated_env: Path, tmp_path: Path
) -> None:
    legacy = isolated_env / ".xpcsjax_xla_mode"
    legacy.write_text("nlsq", encoding="utf-8")
    new_path = tmp_path / "cfg" / "xla_mode"
    new_path.parent.mkdir(parents=True)
    new_path.write_text("auto", encoding="utf-8")
    pi._migrate_legacy_xla_mode(new_path)
    # Legacy is NOT migrated over an existing new file, and is left intact.
    assert new_path.read_text(encoding="utf-8") == "auto"
    assert legacy.exists()


def test_configure_xla_mode_writes(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    assert pi.configure_xla_mode("nlsq", verbose=True, force=True) is True
    assert pi.get_xla_mode_path().read_text(encoding="utf-8") == "nlsq"


def test_configure_xla_mode_preserves_existing_without_force(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    pi.configure_xla_mode("nlsq", force=True)
    # Without force, an existing value is preserved.
    assert pi.configure_xla_mode("auto", verbose=True, force=False) is True
    assert pi.get_xla_mode_path().read_text(encoding="utf-8") == "nlsq"


def test_configure_xla_mode_overwrites_with_force(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    pi.configure_xla_mode("nlsq", force=True)
    assert pi.configure_xla_mode("auto", force=True) is True
    assert pi.get_xla_mode_path().read_text(encoding="utf-8") == "auto"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("nlsq", "nlsq"),
        ("4", "4"),
        ("  8  ", "8"),
        ("garbage", "auto"),
        ("", "auto"),
    ],
)
def test_validate_xla_mode(raw: str, expected: str) -> None:
    assert pi._validate_xla_mode(raw) == expected


# ---------------------------------------------------------------------------
# main() CLI dispatch
# ---------------------------------------------------------------------------


def test_main_skip_both_returns_zero() -> None:
    # Both subsystems skipped -> non-interactive, no work, success.
    assert pi.main(["--no-completion", "--no-xla"]) == 0


def test_main_invalid_xla_mode_normalized(
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    rc = pi.main(["--no-completion", "--xla-mode", "garbage"])
    out = capsys.readouterr().out
    assert "Invalid --xla-mode" in out
    assert rc in (0, 1)
    # Normalized to auto and written.
    assert pi.get_xla_mode_path().read_text(encoding="utf-8") == "auto"


def test_main_noninteractive_full_run(
    monkeypatch: pytest.MonkeyPatch, fake_venv: Path, isolated_env: Path
) -> None:
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: True)
    monkeypatch.setattr(pi, "get_venv_path", lambda: fake_venv)
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_venv))
    rc = pi.main(["--shell", "bash", "--xla-mode", "nlsq", "--verbose"])
    assert rc == 0
    assert (fake_venv / "etc" / "bash_completion.d" / "xpcsjax-completion.sh").is_file()
    assert pi.XLA_BEGIN_MARKER in (fake_venv / "bin" / "activate").read_text("utf-8")


def test_main_dispatches_to_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(pi, "interactive_setup", lambda: called.append(True))
    assert pi.main([]) == 0
    assert called == [True]


# ---------------------------------------------------------------------------
# interactive_setup()
# ---------------------------------------------------------------------------


def test_interactive_setup_aborts_outside_venv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pi, "detect_shell_type", lambda: "bash")
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: False)
    monkeypatch.setattr(pi, "is_conda_environment", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    pi.interactive_setup()
    assert "Aborted." in capsys.readouterr().out


def test_interactive_setup_full_flow(
    monkeypatch: pytest.MonkeyPatch,
    fake_venv: Path,
    isolated_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(pi, "detect_shell_type", lambda: "bash")
    monkeypatch.setattr(pi, "is_virtual_environment", lambda: True)
    monkeypatch.setattr(pi, "is_conda_environment", lambda: False)
    monkeypatch.setattr(pi, "get_venv_path", lambda: fake_venv)
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_venv))

    # Prompts in order: install completion? / XLA mode / add XLA activation?
    answers = iter(["y", "nlsq", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    pi.interactive_setup()
    out = capsys.readouterr().out
    assert "Setup complete!" in out
    assert (fake_venv / "etc" / "bash_completion.d" / "xpcsjax-completion.sh").is_file()
    assert pi.get_xla_mode_path().read_text(encoding="utf-8") == "nlsq"
