xpcsjax.cli
===========

The command-line interface. This subpackage is **lazy-loaded** — importing
:mod:`xpcsjax.cli` does not pull in JAX or any heavy submodule; attribute
access triggers the real import via ``__getattr__``. The console scripts are
declared in ``pyproject.toml`` under ``[project.scripts]``; see
:doc:`/user_guide/cli` for the end-user command reference.

.. currentmodule:: xpcsjax.cli

Main entry point
----------------

``xpcsjax`` (alias ``xj``) is a single flat argument parser — analysis mode is
selected by flags, not by git-style subcommands. The ``--plot-experimental-data``
and ``--plot-simulated-data`` flags switch the run into a standalone-plot path
that skips optimisation.

.. autofunction:: xpcsjax.cli.main.main

.. autofunction:: xpcsjax.cli.main.main_xjexp

.. autofunction:: xpcsjax.cli.main.main_xjsim

The package-level ``xpcsjax/__init__.py`` configures ``JAX_ENABLE_X64``,
``XLA_FLAGS``, and ``NLSQ_SKIP_GPU_CHECK`` *before* the first JAX import.
:func:`~xpcsjax.cli.main.main` does **not** duplicate that setup, but it does
honour ``--threads`` / ``--no-jit`` by injecting them into ``XLA_FLAGS``
ahead of the package-level JAX import.

Argument parsing
----------------

.. automodule:: xpcsjax.cli.args_parser
   :members:

Command dispatch
----------------

.. automodule:: xpcsjax.cli.commands
   :members:

Pipeline stages
---------------

The fit command is split into focused stage modules, each driving one phase of
the ``load → optimise → save → plot`` pipeline:

.. automodule:: xpcsjax.cli.config_handling
   :members:

.. automodule:: xpcsjax.cli.data_pipeline
   :members:

.. automodule:: xpcsjax.cli.optimization_runner
   :members:

.. automodule:: xpcsjax.cli.result_saving
   :members:

.. automodule:: xpcsjax.cli.plot_dispatch
   :members:

Config generator
----------------

The ``xpcsjax-config`` console script emits a populated YAML configuration
from one of the four mode-specific templates, can print a template to stdout
(``--show-template``), validate an existing config (``--validate``), or build
one interactively (``--interactive``).

.. autofunction:: xpcsjax.cli.config_generator.main

.. autofunction:: xpcsjax.cli.config_generator.generate_config

.. autofunction:: xpcsjax.cli.config_generator.show_template

.. autofunction:: xpcsjax.cli.config_generator.validate_config

.. autofunction:: xpcsjax.cli.config_generator.get_template_path

XLA configuration helper
------------------------

``xpcsjax.cli.xla_config`` is dual-use. Imported as a library
(``from xpcsjax.cli.xla_config import configure_xla``) *before* the first
``import xpcsjax``, it overrides the XLA defaults the package would otherwise
set — JAX reads ``XLA_FLAGS`` exactly once at backend init, so ordering
matters. Run as the ``xpcsjax-config-xla`` console script, it is informational
(JAX is already initialised by then).

.. autofunction:: xpcsjax.cli.xla_config.configure_xla

.. autofunction:: xpcsjax.cli.xla_config.get_cpu_info

.. autofunction:: xpcsjax.cli.xla_config.main

Install / uninstall helpers
---------------------------

``xpcsjax-post-install`` (alias ``xj-post-install``) installs shell completion
(bash/zsh/fish), XLA activation scripts, and virtual-environment integration.
``xpcsjax-cleanup`` (alias ``xj-cleanup``) removes everything the post-install
step created.

.. autofunction:: xpcsjax.post_install.main

.. autofunction:: xpcsjax.post_install.interactive_setup

.. autofunction:: xpcsjax.post_install.install_shell_completion

.. autofunction:: xpcsjax.uninstall_scripts.main

.. autofunction:: xpcsjax.uninstall_scripts.interactive_cleanup

.. autofunction:: xpcsjax.uninstall_scripts.find_cleanup_targets
