"""Declarative registry mapping console commands to their completion data.

Single source of truth for the completion generator: which parser backs each
command, what completion-function name it binds to, and the per-command dynamic
hints that argparse cannot express (file/dir intent, free-form suggestions).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field

from xpcsjax import post_install, uninstall_scripts
from xpcsjax.cli import args_parser, config_generator, xla_config
from xpcsjax.gui import app as gui_app
from xpcsjax.runtime.utils import system_validator

# A hint is either a named kind or an explicit tuple of literal completion words.
Hint = str | tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec:
    """One completion function and the commands bound to it.

    Parameters
    ----------
    completion_func : str
        Name of the shell function emitted by the generator.
    command_names : tuple[str, ...]
        All console-script aliases that bind to this function.
    parser_factory : Callable[[], argparse.ArgumentParser]
        Zero-argument callable returning the parser for this command.
    dynamic_hints : dict[str, Hint]
        Mapping from flag string to hint kind or literal word tuple.
        Flags not present here fall back to generic filename completion.
    """

    completion_func: str
    command_names: tuple[str, ...]
    parser_factory: Callable[[], argparse.ArgumentParser]
    dynamic_hints: dict[str, Hint] = field(default_factory=dict)


COMMAND_SPECS: list[CommandSpec] = [
    CommandSpec(
        completion_func="_xpcsjax",
        command_names=("xpcsjax", "xj", "xjexp", "xjsim"),
        parser_factory=args_parser.build_parser,
        dynamic_hints={
            "--config": "configfile", "-c": "configfile",
            "--output": "dir", "-o": "dir",
            "--threads": "threads",
        },
    ),
    CommandSpec(
        completion_func="_xpcsjax_config",
        command_names=("xpcsjax-config", "xj-config"),
        parser_factory=config_generator.build_parser,
        dynamic_hints={
            "--output": "file", "-o": "file",
            "--data": "file", "-d": "file",
        },
    ),
    CommandSpec(
        completion_func="_xpcsjax_config_xla",
        command_names=("xpcsjax-config-xla", "xj-config-xla"),
        parser_factory=xla_config.build_parser,
        dynamic_hints={"--threads": "threads"},
    ),
    CommandSpec(
        completion_func="_xpcsjax_post_install",
        command_names=("xpcsjax-post-install", "xj-post-install"),
        parser_factory=post_install.build_parser,
        dynamic_hints={"--xla-mode": ("auto", "nlsq")},
    ),
    CommandSpec(
        completion_func="_xpcsjax_cleanup",
        command_names=("xpcsjax-cleanup", "xj-cleanup"),
        parser_factory=uninstall_scripts.build_parser,
    ),
    CommandSpec(
        completion_func="_xpcsjax_validate",
        command_names=("xpcsjax-validate", "xj-validate"),
        parser_factory=system_validator.build_parser,
    ),
    CommandSpec(
        completion_func="_xpcsjax_gui",
        command_names=("xpcsjax-gui", "xj-gui"),
        parser_factory=gui_app.build_parser,
    ),
]

__all__ = ["Hint", "CommandSpec", "COMMAND_SPECS"]
