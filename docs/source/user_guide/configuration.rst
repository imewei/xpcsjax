Configuration
=============

.. currentmodule:: xpcsjax


xpcsjax is configuration-driven. Almost every adjustable knob — analysis
mode, parameter bounds, initial values, optimisation strategy, output
options — is set in a single YAML file. The runtime entry point is
:class:`xpcsjax.config.ConfigManager`, which loads, validates, and exposes the
configuration to the rest of the pipeline.

The YAML schema
---------------

A minimal homodyne configuration has five top-level sections:

.. code-block:: yaml

   analysis_mode: static_isotropic

   experimental_data:
     data_file_name: experiment.h5
     # ... beamline-specific fields ...

   analyzer_parameters:
     temporal:
       dt: 0.05
       start_frame: 0
       end_frame: 1000
     scattering:
       wavevector_q: 0.01
     geometry:
       stator_rotor_gap: 1.0e-3

   initial_parameters:
     values: [1.0e3, 0.0, 0.0]    # mode-dependent length

   parameter_bounds:
     D0: [1.0e1, 1.0e5]
     alpha: [-1.0, 1.0]
     D_offset: [-1.0e3, 1.0e3]

   optimization:
     nlsq:
       max_iterations: 1000
       tolerance: 1.0e-8

The same five top-level sections appear in heterodyne configurations,
with two differences: ``analysis_mode`` is ``two_component`` or
``heterodyne``, and the parameter lists under ``initial_parameters`` /
``parameter_bounds`` cover the fourteen heterodyne physics parameters
plus the two contrast/offset scaling parameters. See
:doc:`/user_guide/analysis_modes` for the parameter inventory of each
mode.

Top-level keys
~~~~~~~~~~~~~~

``analysis_mode``
    String selecting the physics model. One of ``"static_anisotropic"``,
    ``"static_isotropic"``, ``"laminar_flow"``, ``"two_component"``
    (or its synonym ``"heterodyne"``). Drives all dispatch in
    :func:`xpcsjax.optimization.nlsq.fit_nlsq` and decides which parameter
    registry is used.

``experimental_data``
    Beamline-shaped block describing the source HDF5 file and any
    dataset path overrides. Consumed by the data loader; see
    :doc:`/user_guide/data_loading`.

``analyzer_parameters``
    Physical constants and analysis-window settings. The ``temporal``
    sub-block (``dt``, ``start_frame``, ``end_frame``) defines the time
    axis. ``scattering.wavevector_q`` is the canonical :math:`q` used
    when the data does not carry per-angle q values. ``geometry``
    holds setup-specific quantities such as the stator–rotor gap for
    laminar-flow analyses.

``initial_parameters``
    A ``values`` list of starting points for the optimiser; the length
    must match the active parameter count for the configured mode.

``parameter_bounds``
    Per-parameter ``[lower, upper]`` pairs. Bounds are enforced inside
    the trust-region solve via the xpcsjax parameter-transform layer
    (see :doc:`/user_guide/nlsq_fitting`).

``optimization.nlsq``
    Optimiser-specific knobs: iteration cap, convergence tolerances,
    strategy hints, anti-degeneracy controller settings, multistart
    configuration. Sensible defaults apply when fields are omitted.

Beyond these five sections, additional blocks (logging, diagnostics,
caching) are recognised and merged in by :class:`xpcsjax.config.ConfigManager`. The
schema is checked at load time; unknown keys are not silently dropped
but they also do not abort the load if they live under namespaces
reserved for forward compatibility.

The :class:`xpcsjax.config.ConfigManager` class
--------------------------------

The high-level entry point is:

.. code-block:: python

   from xpcsjax import ConfigManager

   cfg = ConfigManager("xpcs_config.yaml")
   cfg.load_config()
   loaded = cfg.get_config()
   print(loaded["analysis_mode"])

Constructor
~~~~~~~~~~~

.. code-block:: python

   ConfigManager(
       config_file: str = "xpcsjax_config.yaml",
       config_override: dict | None = None,
   )

``config_file``
    Path to the YAML or JSON configuration file.

``config_override``
    Optional dictionary merged into the loaded configuration after the
    file is read. Used for programmatic overrides — for example, to
    sweep an initial value or a bound without editing the YAML on
    disk.

Public methods
~~~~~~~~~~~~~~

:meth:`xpcsjax.config.ConfigManager.load_config`
    Reads the file (and applies the override), validates the result,
    and stores it on the ``config`` attribute. Returns the loaded
    ``dict``.

:meth:`xpcsjax.config.ConfigManager.get_config`
    Returns the loaded configuration ``dict``. Equivalent to reading
    the ``config`` attribute, but explicit at call sites.

:meth:`xpcsjax.config.ConfigManager.get_model`
    Returns the analysis-mode string. Equivalent to
    ``cfg.config["analysis_mode"]`` but raises a clear error if the
    config has not yet been loaded.

:meth:`xpcsjax.config.ConfigManager.get_target_angle_ranges`
    Returns the phi-angle ranges to use for anisotropic analyses. For
    isotropic and static modes this is typically ``None`` or an
    "all-angles" sentinel.

:meth:`xpcsjax.config.ConfigManager.get_parameter_bounds`
    Returns the bounds ``(lower, upper)`` arrays for the active
    parameters of the configured mode.

:meth:`xpcsjax.config.ConfigManager.get_active_parameters`
    Returns the ordered list of parameter names for the active analysis
    mode, drawn from the parameter registry. The length of this list
    is the dimension of the optimisation problem.

:meth:`xpcsjax.config.ConfigManager.get_initial_parameters`
    Returns the initial-value vector aligned with
    :meth:`xpcsjax.config.ConfigManager.get_active_parameters`.

Direct attribute access
~~~~~~~~~~~~~~~~~~~~~~~

After :meth:`xpcsjax.config.ConfigManager.load_config` has been called the entire configuration is
available as ``ConfigManager.config``. Treat this as read-only; if
you need to mutate values, construct a new manager with
``config_override``.

Programmatic overrides
----------------------

The ``config_override`` argument is a shallow-merged ``dict`` applied
on top of the on-disk YAML. It is the supported way to script
parameter sweeps:

.. code-block:: python

   from xpcsjax import ConfigManager

   for q in [0.005, 0.01, 0.02, 0.04]:
       override = {
           "analyzer_parameters": {"scattering": {"wavevector_q": q}}
       }
       cfg = ConfigManager("base_config.yaml", config_override=override)
       cfg.load_config()
       # ... pass cfg straight through to fit_nlsq ...

Override merging is performed key by key down the tree. To replace an
entire sub-block (rather than merging into it), set the sub-block to
``None`` in the override and then re-populate it.

How :func:`xpcsjax.optimization.nlsq.fit_nlsq` consumes the configuration
-------------------------------------------------------

:func:`xpcsjax.optimization.nlsq.fit_nlsq` accepts the configuration in three forms:

1. A path to a YAML or JSON file (``str`` or ``pathlib.Path``).
2. A pre-built :class:`xpcsjax.config.ConfigManager` instance.
3. A bare ``dict`` (mostly used in tests).

Internally the function extracts ``analysis_mode`` first and dispatches
to either the homodyne or the heterodyne entry point. The selected
analysis mode determines:

* Which physics model is instantiated
  (:class:`xpcsjax.core.HomodyneModel` or the heterodyne stateful model).
* Which parameter registry is consulted for bounds and active-parameter
  ordering.
* Which result type is returned
  (:class:`xpcsjax.optimization.nlsq.results.OptimizationResult` for homodyne, ``list[NLSQResult]``
  for heterodyne).

Because the dispatch happens at fit time, you can hold one
:class:`xpcsjax.config.ConfigManager` instance and call :func:`xpcsjax.optimization.nlsq.fit_nlsq` on it
repeatedly with different ``data`` dictionaries; the manager itself
does not own any solver state.

Validation behaviour
--------------------

:meth:`xpcsjax.config.ConfigManager.load_config` validates:

* Presence of the mandatory top-level keys.
* Consistency between ``analysis_mode`` and the lengths of
  ``initial_parameters.values`` and ``parameter_bounds``.
* Numeric plausibility of bounds (``lower < upper``, finite values).
* Physical plausibility of analyzer parameters (e.g. positive ``dt``,
  ``start_frame < end_frame``).

Validation failures raise concrete exceptions with messages that name
the offending key path. Treat them as errors, not warnings —
proceeding past a validation failure is not supported.

What to read next
-----------------

* :doc:`/user_guide/analysis_modes` for the per-mode parameter
  inventory referenced by ``initial_parameters`` and
  ``parameter_bounds``.
* :doc:`/user_guide/nlsq_fitting` for the ``optimization.nlsq``
  sub-keys.
