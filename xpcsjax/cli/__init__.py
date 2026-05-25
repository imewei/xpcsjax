"""Command-line interface for xpcsjax (NLSQ-only XPCS analysis).

This subpackage is lazy-loaded: importing :mod:`xpcsjax.cli` does NOT
import JAX or any of the heavy submodules. Attribute access (e.g.
``xpcsjax.cli.main``) triggers the actual import via ``__getattr__``.

Mirrors heterodyne's CLI lazy-import surface so test mocks can target
the same import paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# TYPE_CHECKING block: gives Pyright / mypy / IDEs static visibility of the
# lazy-exported symbols without paying the import cost at runtime. The
# ``__getattr__`` below is what actually resolves them at first access.
if TYPE_CHECKING:
    from xpcsjax.cli.args_parser import create_parser, validate_args
    from xpcsjax.cli.commands import dispatch_command
    from xpcsjax.cli.config_generator import main as config_main
    from xpcsjax.cli.config_handling import apply_cli_overrides, load_and_merge_config
    from xpcsjax.cli.data_pipeline import load_and_validate_data, resolve_phi_angles
    from xpcsjax.cli.main import main
    from xpcsjax.cli.optimization_runner import run_nlsq
    from xpcsjax.cli.plot_dispatch import dispatch_plots
    from xpcsjax.cli.xla_config import configure_xla


_IMPORTS: dict[str, tuple[str, str]] = {
    "main": ("xpcsjax.cli.main", "main"),
    "config_main": ("xpcsjax.cli.config_generator", "main"),
    "configure_xla": ("xpcsjax.cli.xla_config", "configure_xla"),
    "create_parser": ("xpcsjax.cli.args_parser", "create_parser"),
    "validate_args": ("xpcsjax.cli.args_parser", "validate_args"),
    "dispatch_command": ("xpcsjax.cli.commands", "dispatch_command"),
    "load_and_merge_config": (
        "xpcsjax.cli.config_handling",
        "load_and_merge_config",
    ),
    "apply_cli_overrides": (
        "xpcsjax.cli.config_handling",
        "apply_cli_overrides",
    ),
    "load_and_validate_data": (
        "xpcsjax.cli.data_pipeline",
        "load_and_validate_data",
    ),
    "resolve_phi_angles": ("xpcsjax.cli.data_pipeline", "resolve_phi_angles"),
    "run_nlsq": ("xpcsjax.cli.optimization_runner", "run_nlsq"),
    "dispatch_plots": ("xpcsjax.cli.plot_dispatch", "dispatch_plots"),
}


def __getattr__(name: str) -> Any:
    if name in _IMPORTS:
        module_path, attr = _IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "main",
    "config_main",
    "configure_xla",
    "create_parser",
    "validate_args",
    "dispatch_command",
    "load_and_merge_config",
    "apply_cli_overrides",
    "load_and_validate_data",
    "resolve_phi_angles",
    "run_nlsq",
    "dispatch_plots",
]
