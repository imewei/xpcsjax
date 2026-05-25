"""Tests for xpcsjax.runtime.shell and the runtime package re-exports."""

from __future__ import annotations

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
