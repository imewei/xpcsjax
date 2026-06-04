"""Post-installation setup for xpcsjax package.

This module provides interactive setup for:
- Shell completion installation (bash/zsh/fish)
- XLA_FLAGS configuration
- Virtual environment integration

CLI Entry Point: xpcsjax-post-install
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Literal

# Block markers injected into venv activate scripts. These are matched
# verbatim by ``uninstall_scripts._remove_xpcsjax_blocks``.
COMPLETION_BEGIN_MARKER = "# >>> xpcsjax completion >>>"
COMPLETION_END_MARKER = "# <<< xpcsjax completion <<<"
XLA_BEGIN_MARKER = "# >>> xpcsjax xla_config >>>"
XLA_END_MARKER = "# <<< xpcsjax xla_config <<<"


def detect_shell_type() -> Literal["bash", "zsh", "fish", "unknown"]:
    """Detect the current shell type.

    Returns:
        Shell type string or "unknown" if detection fails.
    """
    # Check SHELL environment variable
    shell_path = os.environ.get("SHELL", "")
    shell_name = os.path.basename(shell_path)

    if "zsh" in shell_name:
        return "zsh"
    elif "bash" in shell_name:
        return "bash"
    elif "fish" in shell_name:
        return "fish"

    # Fallback: check parent process name
    try:
        import psutil

        parent = psutil.Process().parent()
        if parent:
            pname = parent.name().lower()
            if "zsh" in pname:
                return "zsh"
            elif "bash" in pname:
                return "bash"
            elif "fish" in pname:
                return "fish"
    except (ImportError, OSError, AttributeError):
        pass

    return "unknown"


def is_virtual_environment() -> bool:
    """Check if running in a virtual environment.

    Returns:
        True if in a venv, conda env, or similar.
    """
    # Standard venv check
    if sys.prefix != sys.base_prefix:
        return True

    # Conda environment check
    if os.environ.get("CONDA_PREFIX"):
        return True

    # Check for VIRTUAL_ENV marker
    if os.environ.get("VIRTUAL_ENV"):
        return True

    return False


def is_conda_environment() -> bool:
    """Check if running in a conda/mamba environment.

    Returns:
        True if in a conda environment.
    """
    return bool(os.environ.get("CONDA_PREFIX"))


def get_venv_path() -> Path:
    """Get the virtual environment path.

    Returns:
        Path to the virtual environment directory.
    """
    # Prefer VIRTUAL_ENV if set
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv)

    # Conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix)

    # Fallback to sys.prefix
    return Path(sys.prefix)


def get_completion_source_path() -> Path:
    """Get the path to the completion script in the package.

    Returns:
        Path to completion.sh in the installed package.
    """
    try:
        from xpcsjax.runtime.shell import COMPLETION_SCRIPT

        return COMPLETION_SCRIPT
    except ImportError:
        # Fallback: find relative to this file
        return Path(__file__).parent / "runtime" / "shell" / "completion.sh"


def get_xla_config_source_path(shell: str) -> Path:
    """Get the path to the XLA config script.

    Args:
        shell: Shell type ("bash", "zsh", or "fish")

    Returns:
        Path to the XLA config script.
    """
    try:
        from xpcsjax.runtime.shell import XLA_CONFIG_BASH, XLA_CONFIG_FISH

        if shell == "fish":
            return XLA_CONFIG_FISH
        return XLA_CONFIG_BASH
    except ImportError:
        # Fallback
        base = Path(__file__).parent / "runtime" / "shell" / "activation"
        if shell == "fish":
            return base / "xla_config.fish"
        return base / "xla_config.bash"


def install_bash_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install bash completion script.

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    source = get_completion_source_path()
    if not source.exists():
        if verbose:
            print(f"Completion script not found: {source}")
        return False

    # Install to venv/etc/bash_completion.d/xpcsjax-completion.sh
    dest_dir = venv_path / "etc" / "bash_completion.d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "xpcsjax-completion.sh"

    try:
        shutil.copy2(source, dest)
        if verbose:
            print(f"Installed bash completion to: {dest}")
        return True
    except (OSError, shutil.Error) as e:
        if verbose:
            print(f"Failed to install bash completion: {e}")
        return False


def install_zsh_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install zsh completion script.

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    source = get_completion_source_path()
    if not source.exists():
        if verbose:
            print(f"Completion script not found: {source}")
        return False

    # Install raw completion script to venv/etc/zsh/xpcsjax-completion.sh
    # (zsh wrapper sources it via bashcompinit).
    dest_dir = venv_path / "etc" / "zsh"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "xpcsjax-completion.sh"

    try:
        # Ensure the bash completion is installed first (prerequisite)
        installed_bash = venv_path / "etc" / "bash_completion.d" / "xpcsjax-completion.sh"
        if not installed_bash.exists():
            install_bash_completion(venv_path, verbose=False)

        # Use $VIRTUAL_ENV/$CONDA_PREFIX so paths resolve correctly at
        # activation time and survive venv relocations.
        content = """# Zsh completion for xpcsjax (generated)
# Source the bash completion in zsh-compatible mode

# Ensure completion system is initialized (may already be loaded from .zshrc)
if ! type compdef >/dev/null 2>&1; then
    autoload -Uz compinit
    compinit -C 2>/dev/null
fi
autoload -Uz bashcompinit
bashcompinit

source "${VIRTUAL_ENV:-${CONDA_PREFIX}}/etc/bash_completion.d/xpcsjax-completion.sh" 2>/dev/null
true  # Ensure zero exit code regardless of completion system state
"""
        dest.write_text(content, encoding="utf-8")
        if verbose:
            print(f"Installed zsh completion to: {dest}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to install zsh completion: {e}")
        return False


def install_fish_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install fish completion (basic support).

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    # Fish completions go to a specific location
    dest_dir = venv_path / "share" / "fish" / "vendor_completions.d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "xpcsjax.fish"

    try:
        content = """# Fish completion for xpcsjax (generated)

# xpcsjax
complete -c xpcsjax -s c -l config -d 'Configuration file' -F
complete -c xpcsjax -l mode -d 'Analysis mode' -a 'static_anisotropic static_isotropic laminar_flow two_component'
complete -c xpcsjax -s o -l output -d 'Output directory' -F
complete -c xpcsjax -s v -l verbose -d 'Verbose output'
complete -c xpcsjax -s q -l quiet -d 'Quiet output'
complete -c xpcsjax -l log-level -d 'Log level' -a 'DEBUG INFO WARNING ERROR'
complete -c xpcsjax -s h -l help -d 'Show help'
complete -c xpcsjax -l version -d 'Show version'

# xpcsjax-config
complete -c xpcsjax-config -s o -l output -d 'Output file' -F
complete -c xpcsjax-config -s d -l data -d 'Data file path' -F
complete -c xpcsjax-config -l q -d 'Wavevector magnitude'
complete -c xpcsjax-config -l dt -d 'Time step'
complete -c xpcsjax-config -l time-length -d 'Number of time points'
complete -c xpcsjax-config -l overwrite -d 'Overwrite existing file'
complete -c xpcsjax-config -l show-template -d 'Print template path'
complete -c xpcsjax-config -s i -l interactive -d 'Interactive config builder'
complete -c xpcsjax-config -s V -l validate -d 'Validate config file'
complete -c xpcsjax-config -l mode -d 'Config mode' -a 'static_anisotropic static_isotropic laminar_flow two_component'
complete -c xpcsjax-config -s h -l help -d 'Show help'

# xpcsjax-post-install
complete -c xpcsjax-post-install -s i -l interactive -d 'Interactive setup'
complete -c xpcsjax-post-install -s s -l shell -d 'Shell type' -a 'bash zsh fish'
complete -c xpcsjax-post-install -l no-completion -d 'Skip shell completion'
complete -c xpcsjax-post-install -l no-xla -d 'Skip XLA configuration'
complete -c xpcsjax-post-install -l xla-mode -d 'XLA mode' -a 'auto nlsq'
complete -c xpcsjax-post-install -s v -l verbose -d 'Verbose output'
complete -c xpcsjax-post-install -s h -l help -d 'Show help'

# xpcsjax-cleanup
complete -c xpcsjax-cleanup -s n -l dry-run -d 'Show what would be removed'
complete -c xpcsjax-cleanup -s f -l force -d 'Force cleanup without confirmation'
complete -c xpcsjax-cleanup -s i -l interactive -d 'Interactive cleanup'
complete -c xpcsjax-cleanup -s v -l verbose -d 'Verbose output'
complete -c xpcsjax-cleanup -s h -l help -d 'Show help'

# xpcsjax-validate
complete -c xpcsjax-validate -s v -l verbose -d 'Verbose output'
complete -c xpcsjax-validate -l json -d 'Output results as JSON'
complete -c xpcsjax-validate -s h -l help -d 'Show help'

# Short-alias console scripts (registered in pyproject.toml [project.scripts]).
# Each alias resolves to the same module entry as its full-name counterpart,
# so we simply wrap their completions to inherit the full-name spec.
complete -c xj -w xpcsjax
complete -c xj-config -w xpcsjax-config
complete -c xj-config-xla -w xpcsjax-config-xla
complete -c xj-post-install -w xpcsjax-post-install
complete -c xj-cleanup -w xpcsjax-cleanup
complete -c xj-validate -w xpcsjax-validate

# Plot-only shortcuts (xjexp / xjsim are registered console scripts that
# inject --plot-experimental-data / --plot-simulated-data; mirror xpcsjax's
# completion surface).
complete -c xjexp -w xpcsjax
complete -c xjsim -w xpcsjax
"""
        dest.write_text(content, encoding="utf-8")
        if verbose:
            print(f"Installed fish completion to: {dest}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to install fish completion: {e}")
        return False


def install_shell_completion(
    shell: str | None = None,
    verbose: bool = False,
) -> bool:
    """Install shell completion for the detected or specified shell.

    Args:
        shell: Shell type or None for auto-detection.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    if not is_virtual_environment():
        if verbose:
            print("Not in a virtual environment, skipping completion install")
        return False

    venv_path = get_venv_path()
    detected_shell = shell or detect_shell_type()

    if detected_shell == "unknown":
        if verbose:
            print("Could not detect shell type, trying bash completion")
        detected_shell = "bash"

    if verbose:
        print(f"Installing {detected_shell} completion to {venv_path}")

    if detected_shell == "zsh":
        return install_zsh_completion(venv_path, verbose)
    elif detected_shell == "fish":
        return install_fish_completion(venv_path, verbose)
    else:
        return install_bash_completion(venv_path, verbose)


def install_completion_activation(
    shell: str | None = None,
    verbose: bool = False,
) -> bool:
    """Add completion sourcing to venv activation script.

    Ensures aliases and tab completion are available immediately on
    ``source activate`` without requiring manual shell init changes.

    Args:
        shell: Shell type or None for auto-detection.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    if not is_virtual_environment():
        if verbose:
            print("Not in a virtual environment, skipping completion activation")
        return False

    venv_path = get_venv_path()
    detected_shell = shell or detect_shell_type()

    if detected_shell in ("bash", "zsh", "unknown"):
        return _install_completion_bash_activation(venv_path, verbose)
    elif detected_shell == "fish":
        return _install_completion_fish_activation(venv_path, verbose)
    else:
        return False


def _install_completion_bash_activation(
    venv_path: Path,
    verbose: bool,
) -> bool:
    """Add completion sourcing to bash/zsh activate script."""
    activate_script = venv_path / "bin" / "activate"
    if not activate_script.exists():
        if verbose:
            print(f"Activate script not found: {activate_script}")
        return False

    content = activate_script.read_text(encoding="utf-8")

    if COMPLETION_BEGIN_MARKER in content:
        if verbose:
            print("Completion activation already installed in activate script")
        return True

    # Use $VIRTUAL_ENV so paths resolve correctly at activation time.
    # Zsh needs bashcompinit wrapper; bash sources completion directly.
    # The '2>/dev/null || true' prevents completion system errors (e.g.,
    # missing compdef in minimal zsh) from breaking 'source activate'.
    addition = f"""
{COMPLETION_BEGIN_MARKER}
if [ -n "$ZSH_VERSION" ] && [ -f "$VIRTUAL_ENV/etc/zsh/xpcsjax-completion.sh" ]; then
    source "$VIRTUAL_ENV/etc/zsh/xpcsjax-completion.sh" 2>/dev/null || true
elif [ -f "$VIRTUAL_ENV/etc/bash_completion.d/xpcsjax-completion.sh" ]; then
    source "$VIRTUAL_ENV/etc/bash_completion.d/xpcsjax-completion.sh"
fi
{COMPLETION_END_MARKER}
"""

    try:
        with open(activate_script, "a", encoding="utf-8") as f:
            f.write(addition)
        if verbose:
            print(f"Added completion activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify activate script: {e}")
        return False


def _install_completion_fish_activation(
    venv_path: Path,
    verbose: bool,
) -> bool:
    """Add completion sourcing to fish activate script."""
    activate_script = venv_path / "bin" / "activate.fish"
    if not activate_script.exists():
        if verbose:
            print(f"Fish activate script not found: {activate_script}")
        return False

    content = activate_script.read_text(encoding="utf-8")

    if COMPLETION_BEGIN_MARKER in content:
        if verbose:
            print("Completion activation already installed in fish activate script")
        return True

    addition = f"""
{COMPLETION_BEGIN_MARKER}
if test -f "$VIRTUAL_ENV/share/fish/vendor_completions.d/xpcsjax.fish"
    source "$VIRTUAL_ENV/share/fish/vendor_completions.d/xpcsjax.fish"
end
{COMPLETION_END_MARKER}
"""

    try:
        with open(activate_script, "a", encoding="utf-8") as f:
            f.write(addition)
        if verbose:
            print(f"Added completion activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify fish activate script: {e}")
        return False


def install_xla_config_scripts(venv_path: Path, verbose: bool = False) -> bool:
    """Copy XLA config scripts into ``venv/etc/xpcsjax/`` and fish vendor_conf.d.

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if at least the bash script was installed.
    """
    success = True

    # Bash
    bash_src = get_xla_config_source_path("bash")
    bash_dest_dir = venv_path / "etc" / "xpcsjax"
    bash_dest = bash_dest_dir / "xla_config.bash"
    if bash_src.exists():
        try:
            bash_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bash_src, bash_dest)
            if verbose:
                print(f"Installed XLA bash config to: {bash_dest}")
        except (OSError, shutil.Error) as e:
            if verbose:
                print(f"Failed to install XLA bash config: {e}")
            success = False
    else:
        if verbose:
            print(f"XLA bash source not found: {bash_src}")
        success = False

    # Fish - install to etc/xpcsjax/ AND share/fish/vendor_conf.d/
    fish_src = get_xla_config_source_path("fish")
    fish_dest_etc = bash_dest_dir / "xla_config.fish"
    fish_dest_vendor_dir = venv_path / "share" / "fish" / "vendor_conf.d"
    fish_dest_vendor = fish_dest_vendor_dir / "xpcsjax-xla.fish"
    if fish_src.exists():
        try:
            bash_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fish_src, fish_dest_etc)
            if verbose:
                print(f"Installed XLA fish config to: {fish_dest_etc}")
        except (OSError, shutil.Error) as e:
            if verbose:
                print(f"Failed to install XLA fish config (etc): {e}")
        try:
            fish_dest_vendor_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fish_src, fish_dest_vendor)
            if verbose:
                print(f"Installed XLA fish vendor_conf.d to: {fish_dest_vendor}")
        except (OSError, shutil.Error) as e:
            if verbose:
                print(f"Failed to install XLA fish vendor_conf.d: {e}")
    elif verbose:
        print(f"XLA fish source not found: {fish_src}")

    return success


def install_xla_activation(
    shell: str | None = None,
    verbose: bool = False,
) -> bool:
    """Install XLA configuration to venv activation script.

    Also installs the XLA config scripts themselves into ``venv/etc/xpcsjax/``.

    The XLA *mode* is not a parameter here: callers select it via
    ``configure_xla_mode`` (which writes the mode file), and the activation
    hook installed by this function sources the config script that reads that
    file at activation time.

    Args:
        shell: Shell type or None for auto-detection.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    if not is_virtual_environment():
        if verbose:
            print("Not in a virtual environment, skipping XLA activation install")
        return False

    venv_path = get_venv_path()
    detected_shell = shell or detect_shell_type()

    # First, install the XLA config script files into the venv.
    install_xla_config_scripts(venv_path, verbose=verbose)

    if detected_shell in ("bash", "zsh", "unknown"):
        return _install_xla_bash_activation(venv_path, verbose)
    elif detected_shell == "fish":
        return _install_xla_fish_activation(venv_path, verbose)
    else:
        return False


def _install_xla_bash_activation(
    venv_path: Path,
    verbose: bool,
) -> bool:
    """Install XLA config to bash/zsh activate script.

    The XLA *mode* is intentionally not threaded here: this only appends a
    ``source`` directive, and the sourced ``xla_config.bash`` reads the mode
    file (written by ``configure_xla_mode``) at activation time.
    """
    activate_script = venv_path / "bin" / "activate"
    if not activate_script.exists():
        if verbose:
            print(f"Activate script not found: {activate_script}")
        return False

    # Check if already installed
    content = activate_script.read_text(encoding="utf-8")

    if XLA_BEGIN_MARKER in content:
        if verbose:
            print("XLA activation already installed in activate script")
        return True

    # Reference the venv-installed XLA config (relocation-safe via
    # $VIRTUAL_ENV / $CONDA_PREFIX).
    xla_script = "${VIRTUAL_ENV:-${CONDA_PREFIX}}/etc/xpcsjax/xla_config.bash"

    addition = f"""
{XLA_BEGIN_MARKER}
if [ -f "{xla_script}" ]; then
    source "{xla_script}"
fi
{XLA_END_MARKER}
"""

    try:
        with open(activate_script, "a", encoding="utf-8") as f:
            f.write(addition)
        if verbose:
            print(f"Added XLA activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify activate script: {e}")
        return False


def _install_xla_fish_activation(
    venv_path: Path,
    verbose: bool,
) -> bool:
    """Install XLA config to fish activate script.

    The XLA *mode* is intentionally not threaded here: this only appends a
    ``source`` directive, and the sourced ``xla_config.fish`` reads the mode
    file (written by ``configure_xla_mode``) at activation time.
    """
    activate_script = venv_path / "bin" / "activate.fish"
    if not activate_script.exists():
        if verbose:
            print(f"Fish activate script not found: {activate_script}")
        return False

    # Check if already installed
    content = activate_script.read_text(encoding="utf-8")

    if XLA_BEGIN_MARKER in content:
        if verbose:
            print("XLA activation already installed in fish activate script")
        return True

    # Resolve the XLA config relative to the active environment at activation
    # time, matching the bash hook and ``get_venv_path()``: prefer
    # ``$VIRTUAL_ENV`` and fall back to ``$CONDA_PREFIX``. fish has no
    # ``${VAR:-default}`` expansion, so the fallback is spelled out. ``test -n``
    # (rather than ``set -q``) treats an empty value as unset, matching the
    # bash ``:-`` semantics. ``set -l`` keeps the temp var scoped to this
    # sourced block so it never leaks into the interactive session.
    addition = f"""
{XLA_BEGIN_MARKER}
set -l __xpcsjax_xla_prefix
if test -n "$VIRTUAL_ENV"
    set __xpcsjax_xla_prefix $VIRTUAL_ENV
else if test -n "$CONDA_PREFIX"
    set __xpcsjax_xla_prefix $CONDA_PREFIX
end
if test -n "$__xpcsjax_xla_prefix"; and test -f "$__xpcsjax_xla_prefix/etc/xpcsjax/xla_config.fish"
    source "$__xpcsjax_xla_prefix/etc/xpcsjax/xla_config.fish"
end
{XLA_END_MARKER}
"""

    try:
        with open(activate_script, "a", encoding="utf-8") as f:
            f.write(addition)
        if verbose:
            print(f"Added XLA activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify fish activate script: {e}")
        return False


def get_xla_mode_path() -> Path:
    """Get the path for the XLA mode configuration file.

    Uses the virtual environment if active, otherwise XDG config directory.
    Priority: $VIRTUAL_ENV or $CONDA_PREFIX > $XDG_CONFIG_HOME/xpcsjax.

    Returns:
        Path to the XLA mode file.
    """
    # Prefer per-environment config
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if venv:
        return Path(venv) / "etc" / "xpcsjax" / "xla_mode"

    # Fall back to XDG config directory
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
    if not xdg_config:
        xdg_config = str(Path.home() / ".config")
    return Path(xdg_config) / "xpcsjax" / "xla_mode"


def _migrate_legacy_xla_mode(new_path: Path) -> None:
    """Migrate legacy ~/.xpcsjax_xla_mode to new location if it exists."""
    legacy = Path.home() / ".xpcsjax_xla_mode"
    if legacy.exists() and not new_path.exists():
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            mode = legacy.read_text(encoding="utf-8").strip()
            new_path.write_text(mode, encoding="utf-8")
            legacy.unlink()
        except OSError:
            pass  # Best-effort migration


def configure_xla_mode(mode: str = "auto", verbose: bool = False, force: bool = False) -> bool:
    """Configure the XLA mode.

    Stores in the virtual environment (if active) or XDG config directory.
    Preserves existing configuration unless ``force=True``, matching the
    shell-side behavior in ``xla_config.bash`` where the mode file is only
    written when an explicit argument is provided.

    Args:
        mode: XLA mode (auto, nlsq, or a number).
        verbose: Print verbose output.
        force: Overwrite existing config. Set to True only when the user
            explicitly passes ``--xla-mode``.

    Returns:
        True if configuration succeeded.
    """
    config_file = get_xla_mode_path()

    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Migrate legacy file if new file doesn't exist yet
        _migrate_legacy_xla_mode(config_file)

        if config_file.exists() and not force:
            if verbose:
                existing = config_file.read_text(encoding="utf-8").strip()
                print(f"Preserving existing XLA mode '{existing}' in {config_file}")
            return True

        config_file.write_text(mode, encoding="utf-8")
        if verbose:
            print(f"Set XLA mode to '{mode}' in {config_file}")

        # Clean up legacy file if it exists
        legacy = Path.home() / ".xpcsjax_xla_mode"
        if legacy.exists():
            legacy.unlink(missing_ok=True)
            if verbose:
                print(f"Removed legacy config: {legacy}")

        return True
    except OSError as e:
        if verbose:
            print(f"Failed to write XLA mode config: {e}")
        return False


def _validate_xla_mode(mode: str) -> str:
    """Validate an XLA mode string. Accepts 'auto', 'nlsq', or an integer.

    Returns the normalized mode string, or 'auto' on invalid input.
    """
    mode = mode.strip().lower()
    if mode in ("auto", "nlsq"):
        return mode
    try:
        int(mode)
        return mode
    except ValueError:
        return "auto"


def interactive_setup() -> None:
    """Run interactive post-installation setup."""
    print("=" * 60)
    print("xpcsjax Post-Installation Setup")
    print("=" * 60)
    print()

    # Detect environment
    shell = detect_shell_type()
    in_venv = is_virtual_environment()
    is_conda = is_conda_environment()

    print(f"Detected shell: {shell}")
    print(f"Virtual environment: {in_venv}")
    if is_conda:
        print(f"Conda environment: {os.environ.get('CONDA_PREFIX', '')}")
    elif in_venv:
        print(f"Venv path: {get_venv_path()}")
    print()

    if not in_venv:
        print("WARNING: Not running in a virtual environment.")
        print("Shell completion and XLA activation require a virtual environment.")
        print()
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # Shell completion
    print("\n--- Shell Completion ---")
    response = input(f"Install {shell} shell completion? [Y/n]: ").strip().lower()
    if response != "n":
        success = install_shell_completion(shell, verbose=True)
        if success:
            act_success = install_completion_activation(shell, verbose=True)
            if act_success:
                print("Shell completion installed and activated!")
                print("Deactivate and reactivate your venv to load aliases.")
            else:
                print("Shell completion installed (activate hook failed).")
                env_var = "$CONDA_PREFIX" if is_conda else "$VIRTUAL_ENV"
                if shell == "zsh":
                    print(f"Add to ~/.zshrc: source {env_var}/etc/zsh/xpcsjax-completion.sh")
                elif shell == "bash":
                    print(
                        f"Add to ~/.bashrc: source {env_var}/etc/bash_completion.d/xpcsjax-completion.sh"
                    )
        else:
            print("Shell completion installation failed.")
    print()

    # XLA Configuration
    print("\n--- XLA Configuration ---")
    print("XLA modes control how many CPU devices JAX uses:")
    print("  auto    - Auto-detect based on CPU cores (recommended)")
    print("  nlsq    - Single device for NLSQ fitting")
    print("  <N>     - Explicit integer device count")
    print()

    raw_mode = input("Select XLA mode [auto]: ").strip().lower() or "auto"
    mode = _validate_xla_mode(raw_mode)
    if mode != raw_mode:
        print(f"Invalid mode: {raw_mode!r}, using 'auto'")

    success = configure_xla_mode(mode, verbose=True, force=True)
    if success:
        print(f"XLA mode set to '{mode}'")

    # Install XLA activation
    response = input("\nAdd XLA config to venv activate script? [Y/n]: ").strip().lower()
    if response != "n":
        success = install_xla_activation(shell, verbose=True)
        if success:
            print("XLA activation installed!")
            print("Deactivate and reactivate your venv to apply.")
        else:
            print("XLA activation installation failed.")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("To verify installation, run: xpcsjax-validate")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for xpcsjax-post-install."""
    parser = argparse.ArgumentParser(
        description="Post-installation setup for xpcsjax",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xpcsjax-post-install                  # Interactive setup
  xpcsjax-post-install --shell zsh      # Install zsh completion
  xpcsjax-post-install --no-xla         # Skip XLA configuration
""",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive setup (default if no options)",
    )
    parser.add_argument(
        "--shell",
        "-s",
        choices=["bash", "zsh", "fish"],
        help="Shell type for completion installation",
    )
    parser.add_argument(
        "--no-completion",
        action="store_true",
        help="Skip shell completion installation",
    )
    parser.add_argument(
        "--no-xla",
        action="store_true",
        help="Skip XLA configuration",
    )
    parser.add_argument(
        "--xla-mode",
        type=str,
        default=None,
        help="XLA configuration mode: 'auto', 'nlsq', or an integer device count "
        "(default: preserve existing, or auto)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args(argv)

    # Validate --xla-mode if provided
    if args.xla_mode is not None:
        normalized = _validate_xla_mode(args.xla_mode)
        if normalized != args.xla_mode.strip().lower():
            print(
                f"Invalid --xla-mode {args.xla_mode!r}; expected 'auto', 'nlsq', or "
                f"an integer. Falling back to 'auto'."
            )
        args.xla_mode = normalized

    # Run interactive setup if no specific options given
    if args.interactive or (
        not args.no_completion and not args.no_xla and not args.shell and not args.xla_mode
    ):
        interactive_setup()
        return 0

    # Non-interactive mode
    success = True

    if not args.no_completion:
        result = install_shell_completion(args.shell, args.verbose)
        if not result:
            print("Shell completion installation failed")
            success = False
        else:
            result = install_completion_activation(args.shell, args.verbose)
            if not result:
                print("Completion activation hook failed")
                success = False

    if not args.no_xla:
        xla_mode = args.xla_mode or "auto"
        xla_explicit = args.xla_mode is not None
        result = configure_xla_mode(xla_mode, args.verbose, force=xla_explicit)
        if not result:
            print("XLA mode configuration failed")
            success = False

        result = install_xla_activation(args.shell, args.verbose)
        if not result:
            print("XLA activation installation failed")
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
