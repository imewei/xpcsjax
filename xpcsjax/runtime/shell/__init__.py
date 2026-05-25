"""Shell completion and activation scripts for xpcsjax.

Contents:
    * ``completion.sh`` — bash/zsh completion for the xpcsjax CLI commands.
    * ``activation/xla_config.bash`` — XLA_FLAGS auto-config for bash/zsh.
    * ``activation/xla_config.fish`` — XLA_FLAGS auto-config for fish.

These files are installed into the active virtualenv by
``xpcsjax.post_install`` and sourced from the venv's activate scripts.
"""

from __future__ import annotations

from pathlib import Path

SHELL_DIR = Path(__file__).parent
COMPLETION_SCRIPT = SHELL_DIR / "completion.sh"
ACTIVATION_DIR = SHELL_DIR / "activation"
XLA_CONFIG_BASH = ACTIVATION_DIR / "xla_config.bash"
XLA_CONFIG_FISH = ACTIVATION_DIR / "xla_config.fish"


def get_completion_script() -> str:
    """Return the absolute path of the bash/zsh completion script."""
    return str(COMPLETION_SCRIPT.resolve())


def get_xla_config_script(shell: str = "bash") -> str:
    """Return the absolute path of the XLA config script for ``shell``.

    Args:
        shell: One of ``"bash"``, ``"zsh"``, ``"fish"``.
    """
    if shell in ("bash", "zsh"):
        return str(XLA_CONFIG_BASH.resolve())
    if shell == "fish":
        return str(XLA_CONFIG_FISH.resolve())
    raise ValueError(f"Unsupported shell: {shell!r}")


__all__ = [
    "SHELL_DIR",
    "COMPLETION_SCRIPT",
    "ACTIVATION_DIR",
    "XLA_CONFIG_BASH",
    "XLA_CONFIG_FISH",
    "get_completion_script",
    "get_xla_config_script",
]
