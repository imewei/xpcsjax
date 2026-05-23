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

   The bare value ``"static"`` is **not accepted** — it was ambiguous
   between the isotropic and anisotropic variants. Configs must
   specify one explicitly. See
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
   mode and silently collapsed to one of these two downstream. It has
   been removed; if you have an old config, replace
   ``analysis_mode: static`` with either ``static_anisotropic`` (the
   safer default — preserves angle resolution) or ``static_isotropic``.

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
* Per-angle scaling parameters that, after Fourier reparameterisation,
  collapse into a small number of Fourier coefficients for multi-angle
  fits.

Each phi-angle stratum is fit jointly. The return type of
:func:`xpcsjax.optimization.nlsq.fit_nlsq` in this mode is ``list[NLSQResult]`` — one
entry per phi-angle group, in the same order as the angle list in the
input data dictionary.

.. note::

   The full sixteen-parameter problem is overdetermined for any single
   phi angle; the multi-angle Fourier reparameterisation is what makes
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

Programmatic mode inspection
----------------------------

You can read the configured mode and its active parameters before any
fit is run:

.. code-block:: python

   from xpcsjax import ConfigManager

   cfg = ConfigManager("xpcs_config.yaml")
   cfg.load_config()

   print(cfg.get_model())                  # e.g. "laminar_flow"
   print(cfg.get_active_parameters())      # e.g. ["D0", "alpha", "D_offset", ...]
   lo, hi = cfg.get_parameter_bounds()
   print(list(zip(cfg.get_active_parameters(), lo, hi)))

This is the recommended sanity check before launching a long fit on a
new configuration.

Where to go next
----------------

* :doc:`/user_guide/homodyne_workflow` for an end-to-end script using
  the three homodyne modes.
* :doc:`/user_guide/heterodyne_workflow` for the two-component
  pipeline and the Fourier reparameterisation.
