"""Tests for xpcsjax.runtime.shell and the runtime package re-exports."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from xpcsjax.runtime.shell import (
    ACTIVATION_DIR,
    COMPLETION_SCRIPT,
    SHELL_DIR,
    XLA_CONFIG_BASH,
    XLA_CONFIG_FISH,
    get_completion_script,
    get_xla_config_script,
)


def test_module_path_constants_point_at_shipped_files() -> None:
    assert SHELL_DIR.is_dir()
    assert COMPLETION_SCRIPT.name == "completion.sh"
    assert COMPLETION_SCRIPT.is_file()
    assert ACTIVATION_DIR.is_dir()
    assert XLA_CONFIG_BASH.is_file()
    assert XLA_CONFIG_FISH.is_file()


def test_get_completion_script_returns_absolute_existing_path() -> None:
    p = get_completion_script()
    assert Path(p).is_absolute()
    assert Path(p).is_file()
    assert p == str(COMPLETION_SCRIPT.resolve())


@pytest.mark.parametrize(
    ("shell", "expected"),
    [
        ("bash", XLA_CONFIG_BASH),
        ("zsh", XLA_CONFIG_BASH),
        ("fish", XLA_CONFIG_FISH),
    ],
)
def test_get_xla_config_script_per_shell(shell: str, expected: Path) -> None:
    assert get_xla_config_script(shell) == str(expected.resolve())


def test_get_xla_config_script_defaults_to_bash() -> None:
    assert get_xla_config_script() == str(XLA_CONFIG_BASH.resolve())


def test_get_xla_config_script_rejects_unknown_shell() -> None:
    with pytest.raises(ValueError, match="Unsupported shell"):
        get_xla_config_script("powershell")


def _source_xla_config(tmp_path: Path, mode: str) -> tuple[int, str, bool]:
    """Source xla_config.bash with ``mode`` in an isolated HOME.

    An empty ``mode`` exercises the no-argument load branch that every real
    shell activation takes.

    Returns (exit status, resulting XLA_FLAGS, whether a mode file was persisted).
    """
    script = 'source "$1" "$2"; echo "rc=$?"; echo "flags=${XLA_FLAGS:-}"'
    proc = subprocess.run(  # noqa: S603
        ["bash", "-c", script, "bash", str(XLA_CONFIG_BASH), mode],  # noqa: S607
        capture_output=True,
        text=True,
        # Deliberately minimal env (isolates from any XLA_FLAGS/xpcsjax state in
        # the parent process) PLUS the handful of Windows-only vars
        # (SystemRoot/ComSpec/TEMP/TMP) that Windows' CreateProcess and bash's
        # own DLL loading commonly need -- a bare PATH/HOME dict silently
        # misbehaves there (empty/garbled subprocess output) even though POSIX
        # doesn't need them.
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path),
            **{
                k: os.environ[k]
                for k in ("SystemRoot", "ComSpec", "TEMP", "TMP")
                if k in os.environ
            },
        },
    )
    out = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    persisted = (tmp_path / "xpcsjax" / "xla_mode").is_file()
    return int(out["rc"]), out["flags"], persisted


def test_xla_config_bash_accepts_integer_mode(tmp_path: Path) -> None:
    rc, flags, persisted = _source_xla_config(tmp_path, "8")
    assert rc == 0
    assert "--xla_force_host_platform_device_count=8" in flags
    assert persisted


@pytest.mark.parametrize("mode", ["8x", "4 5", "0x10"])
def test_xla_config_bash_rejects_digit_prefixed_garbage(tmp_path: Path, mode: str) -> None:
    # A bare `[0-9]*)` case glob matched any digit-prefixed string and persisted it.
    rc, flags, persisted = _source_xla_config(tmp_path, mode)
    assert rc != 0
    assert mode not in flags
    assert not persisted


def test_xla_config_bash_no_arg_uses_saved_mode(tmp_path: Path) -> None:
    _source_xla_config(tmp_path, "2")
    rc, flags, _ = _source_xla_config(tmp_path, "")
    assert rc == 0
    assert "--xla_force_host_platform_device_count=2" in flags


def test_xla_config_bash_no_arg_recovers_from_poisoned_mode_file(tmp_path: Path) -> None:
    # Older releases could persist an invalid mode; activation must not brick.
    mode_file = tmp_path / "xpcsjax" / "xla_mode"
    mode_file.parent.mkdir(parents=True)
    mode_file.write_text("8x\n")

    rc, flags, _ = _source_xla_config(tmp_path, "")
    assert rc == 0
    assert "xla_force_host_platform_device_count=" in flags
    assert "8x" not in flags


def test_runtime_package_reexports() -> None:
    import xpcsjax.runtime as rt

    # The top-level runtime package surface.
    for symbol in (
        "SystemValidator",
        "ValidationResult",
        "Severity",
        "run_validation",
        "get_completion_script",
        "get_xla_config_script",
    ):
        assert hasattr(rt, symbol), symbol


def test_runtime_utils_package_reexports() -> None:
    import xpcsjax.runtime.utils as ru

    for symbol in ("SystemValidator", "ValidationResult", "Severity", "run_validation"):
        assert hasattr(ru, symbol), symbol
