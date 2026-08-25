Analysis modes
==============

.. currentmodule:: xpcsjax


The ``analysis_mode`` top-level key in the YAML configuration selects
which physics model xpcsjax fits. There are four canonical modes,
grouped into two families:

* **Homodyne family.** ``static_anisotropic``, ``static_isotropic``,
  ``laminar_flow``. Backed by :class:`xpcsjax.core.HomodyneModel`.
* **Heterodyne family.** ``two_component`` (with ``heterodyne`` accepted
  as a case-insensitive synonym). Backed by the two-component stateful
  heterodyne model.

.. note::

   The bare value ``"static"`` is a **deprecated alias** — it was
   ambiguous between the isotropic and anisotropic variants.
   ``ConfigManager`` still accepts it but normalises it to
   ``static_anisotropic`` and emits a deprecation warning; set one of the
   canonical modes explicitly to silence it. See
   :doc:`/development/porting_notes` for the migration path.

The choice of mode determines the active parameter count, the
parameter names, the physics kernel used to compute the model
correlation function, and the dispatch path through
:func:`xpcsjax.optimization.nlsq.fit_nlsq`.

Choosing a mode
---------------

The decision usually falls out of the experiment:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Experimental scenario
     - Recommended mode
   * - Equilibrium sample, no angular structure
     - ``static_isotropic``
   * - Equilibrium sample with directional structure
     - ``static_anisotropic``
   * - Sample under laminar shear flow
     - ``laminar_flow``
   * - Two-component (e.g. fluctuating + drifting) dynamics
     - ``two_component`` (or ``heterodyne``)

Static family (3 parameters each)
---------------------------------

The two equilibrium-sample modes share the same physics kernel and the
same three active parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Meaning
   * - ``D0``
     - Effective diffusion coefficient at the reference time.
   * - ``alpha``
     - Time-exponent of the diffusion law,
       :math:`D(t) \propto t^{\alpha}`.
   * - ``D_offset``
     - Additive offset on the diffusion term, absorbing background
       structure.

They differ only in **data preparation**, not in the kernel.

Static isotropic
~~~~~~~~~~~~~~~~

Collapses the phi axis before the residual is computed, fitting against
the angularly-averaged correlation. This is the default mode for
textbook diffusive XPCS analyses where the sample has no directional
structure.

Static anisotropic
~~~~~~~~~~~~~~~~~~

Subsets the phi-angle list according to ``target_angle_ranges`` from
the configuration, then performs a stratified fit that retains angular
resolution. Use this mode when the sample shows directional structure
in :math:`g_2(q, \phi, t)` that an isotropic fit would average away.

.. note::

   Pre-rename, a bare ``"static"`` value was treated as a third static
   mode and silently collapsed to one of these two downstream. It is now
   a deprecated alias: ``ConfigManager`` still accepts
   ``analysis_mode: static`` and automatically maps it to
   ``static_anisotropic`` with a deprecation warning. For clarity, set
   ``static_anisotropic`` (the safer default — preserves angle
   resolution) or ``static_isotropic`` explicitly to silence the warning.

Laminar flow (7 parameters)
---------------------------

The homodyne model under steady laminar shear. Seven active
parameters: the three diffusion terms inherited from the static modes
plus four shear-related quantities (shear rate, geometry, sensitivity
weights). The geometry block in the configuration becomes load-bearing
in this mode — ``analyzer_parameters.geometry.stator_rotor_gap``
participates directly in the model kernel.

The xpcsjax shear-weighting layer (:mod:`xpcsjax.optimization.nlsq.shear_weighting`)
adds an angle-dependent weight to the residuals so that the fit is not
dominated by directions where shear is degenerate.

.. note::

   The seven-parameter laminar-flow model is the most failure-prone
   mode in xpcsjax. The anti-degeneracy controller is on by default;
   do not disable it without first running the multistart pathway.

Two-component and heterodyne (14 physics + 2 scaling)
-----------------------------------------------------

The two-component model (``analysis_mode: two_component``) is the
canonical name; ``heterodyne`` (case-insensitive) and ``two-component``
are accepted synonyms that the config loader normalises to
``two_component`` at load time. Both expose fourteen physics
parameters and two scaling parameters (typically ``contrast`` and
``offset``).

The fourteen physics parameters cover:

* Two diffusive components, each with its own ``D0``, ``alpha``, and
  ``D_offset`` (six parameters).
* Their relative amplitudes / mixing fractions.
* Cross-component coupling and the reference-beam-induced terms.
* Per-angle scaling parameters that, after per-angle reparameterisation,
  collapse into a small, shared set of parameters for multi-angle fits.

Each phi-angle stratum is fit jointly. :func:`xpcsjax.optimization.nlsq.fit_nlsq`
in this mode returns a single
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`; the per-angle
detail (``chi2_per_angle``, ``contrast_per_angle`` / ``offset_per_angle``) is
recorded under ``result.nlsq_diagnostics``, in the same order as the angle list
in the input data dictionary.

.. note::

   The full sixteen-parameter problem is overdetermined for any single
   phi angle; the multi-angle per-angle reparameterisation is what makes
   the fit identifiable in practice. See
   :doc:`/user_guide/heterodyne_workflow` for details.

Parameter inventory matrix
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Mode
     - Param count
     - Active parameter family
   * - ``static_isotropic``
     - 3
     - ``D0``, ``alpha``, ``D_offset``; isotropic data prep
   * - ``static_anisotropic``
     - 3
     - same family; anisotropic data prep + ``target_angle_ranges``
   * - ``laminar_flow``
     - 7
     - diffusion family + 4 shear / geometry terms
   * - ``two_component``
     - 14 + 2
     - two diffusion families + coupling + per-angle scaling

The authoritative active-parameter ordering for any given mode is the
``list[str]`` returned by
:meth:`xpcsjax.config.ConfigManager.get_active_parameters`. Use it to align
the ``values`` array in ``initial_parameters`` and the bounds map in
``parameter_bounds``.

To vary only a subset of a mode's parameters, or freeze one at a constant
without dropping it from the model, see
:ref:`fixed_active_parameters` in :doc:`/user_guide/configuration`.

Programmatic mode inspection
----------------------------

You can read the configured mode and its active parameters before any
fit is run:

.. code-block:: python

   from xpcsjax import ConfigManager

   cfg = ConfigManager("xpcs_config.yaml")
   cfg.load_config()

   print(cfg.analysis_mode)                 # AnalysisMode.LAMINAR_FLOW
   print(cfg.analysis_mode.value)           # "laminar_flow"  (bare string)
   print(cfg.get_active_parameters())       # e.g. ["D0", "alpha", "D_offset", ...]
   lo, hi = cfg.get_parameter_bounds()
   print(list(zip(cfg.get_active_parameters(), lo, hi)))

This is the recommended sanity check before launching a long fit on a
new configuration.

Where to go next
----------------

* :doc:`/user_guide/homodyne_workflow` for an end-to-end script using
  the three homodyne modes.
* :doc:`/user_guide/heterodyne_workflow` for the two-component
  pipeline and the per-angle reparameterisation.
