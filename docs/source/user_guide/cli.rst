Command-line interface
======================

Installing xpcsjax registers a family of console scripts. Every full-name
command has a short ``xj`` alias, so ``xpcsjax`` and ``xj`` are interchangeable.

.. list-table:: Console scripts
   :header-rows: 1
   :widths: 28 22 50

   * - Full name
     - Alias
     - Purpose
   * - ``xpcsjax``
     - ``xj``
     - Run an NLSQ fit (or a standalone QC/simulation plot).
   * - ``xpcsjax-config``
     - ``xj-config``
     - Generate / show / validate a YAML configuration.
   * - ``xpcsjax-config-xla``
     - ``xj-config-xla``
     - Inspect or print the CPU ``XLA_FLAGS`` xpcsjax uses.
   * - ``xpcsjax-validate``
     - ``xj-validate``
     - Validate the installation (env, deps, JAX, templates).
   * - ``xpcsjax-post-install``
     - ``xj-post-install``
     - Install shell completion + XLA activation scripts.
   * - ``xpcsjax-cleanup``
     - ``xj-cleanup``
     - Remove everything ``post-install`` created.
   * - ``xjexp``
     - —
     - Shortcut for plotting experimental data (QC).
   * - ``xjsim``
     - —
     - Shortcut for plotting simulated C₂ heatmaps.

.. note::

   ``xpcsjax`` is a single flat command — the analysis mode is chosen by
   **flags**, not by git-style subcommands. ``xjexp`` and ``xjsim`` are thin
   wrappers that preset the standalone-plot flags (mirroring the upstream
   ``heterodyne`` package's ``hexp`` / ``hsim`` shortcuts).

Running a fit
-------------

The minimal invocation needs only a config file; analysis mode, bounds, and
initial parameters all come from the YAML:

.. code-block:: console

   $ xpcsjax --config config.yaml

Common overrides (all optional; each takes precedence over the YAML):

.. code-block:: console

   $ xpcsjax -c config.yaml \
       --output results/run1 \
       --output-format both \
       --mode static_anisotropic \
       --phi 0,45,90 \
       --multistart --multistart-n 20 \
       --max-iterations 500 --tolerance 1e-8 \
       --threads 8

Key fit flags:

``-c/--config``
   Path to the YAML configuration (required for a fit).
``-o/--output``
   Output directory; overrides ``output_settings.output_dir`` in the YAML.
``--output-format {json,npz,both}``
   Result serialisation format (default ``both``).
``--mode``
   Force the analysis mode (``static_isotropic``, ``static_anisotropic``,
   ``laminar_flow``, ``two_component``); overrides the YAML.
``--phi``
   Comma-separated φ angles in degrees to analyse; overrides the config.
``--multistart`` / ``--no-multistart`` / ``--multistart-n N``
   Enable, disable, or size the Latin-hypercube multistart.
``--max-iterations`` / ``--tolerance``
   Trust-region iteration cap and convergence tolerance.
``--initial-*``
   Per-parameter initial-value overrides (e.g. ``--initial-D0``,
   ``--initial-alpha``, ``--initial-gamma-dot-t0`` for laminar flow, the
   ``--initial-v0`` / ``--initial-f0..f3`` family for ``two_component``).

Execution control:

``--threads N``
   CPU thread count injected into ``XLA_FLAGS`` before the JAX import.
``--no-jit``
   Disable JIT compilation — debugging only, much slower.
``-v/--verbose`` (repeatable), ``-q/--quiet``
   Verbosity controls.

Plotting
--------

By default a fit also generates diagnostic plots (``--plot``; suppress with
``--no-plot``). Plot-specific flags:

``--save-plots``
   Write fit-comparison plots to the output directory.
``--plotting-backend {auto,matplotlib,datashader}``
   Rendering backend (default ``auto``; Datashader is the fast path).
``--parallel-plots``
   Render per-angle plots in parallel via multiprocessing (Datashader path).

Two flags switch into a **standalone plot** that skips optimisation entirely:

.. code-block:: console

   # Quality-control: plot the experimental C₂ surfaces
   $ xpcsjax --config config.yaml --plot-experimental-data
   $ xjexp --config config.yaml          # equivalent shortcut

   # Plot simulated C₂ heatmaps from the config parameters
   $ xpcsjax --config config.yaml --plot-simulated-data --contrast 0.3 --offset-sim 1.0
   $ xjsim --config config.yaml          # equivalent shortcut

Generating a configuration
--------------------------

``xpcsjax-config`` emits a populated YAML from one of the four mode templates:

.. code-block:: console

   # Write a static_anisotropic config wired to a data file
   $ xpcsjax-config --mode static_anisotropic --output config.yaml \
       --data data.h5 --q 0.01 --dt 0.1 --time-length 1000

   # Print a template to stdout without writing
   $ xpcsjax-config --mode laminar_flow --show-template

   # Validate an existing config
   $ xpcsjax-config --mode two_component --output config.yaml --validate

   # Build one interactively
   $ xpcsjax-config --mode static_isotropic --interactive

Use ``--overwrite`` to replace an existing output file.

Validating the installation
---------------------------

.. code-block:: console

   $ xpcsjax-validate

Checks environment detection, dependency versions, JAX/float64 configuration,
and template / public-API integrity. Because xpcsjax is NLSQ-only, the
validator does **not** look for any Bayesian / MCMC dependency.

Shell completion and activation
-------------------------------

``xpcsjax-post-install`` installs shell completion (bash/zsh/fish) and the XLA
activation scripts that export the tuned CPU ``XLA_FLAGS`` on virtual-env
activation:

.. code-block:: console

   $ xpcsjax-post-install

``xpcsjax-cleanup`` reverses it, removing the completion files, XLA config, and
activation-script edits:

.. code-block:: console

   $ xpcsjax-cleanup

See :doc:`/api/cli` and :doc:`/api/runtime` for the importable APIs behind
these scripts.
