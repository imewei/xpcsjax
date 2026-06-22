Heterodyne workflow
===================

.. currentmodule:: xpcsjax


The heterodyne pipeline fits the two-component XPCS model
(``analysis_mode: two_component`` or its alias ``"heterodyne"``)
against multi-angle two-time correlation data. It differs from the
homodyne pipeline in three places:

1. The parameter set is much larger — fourteen physics parameters
   plus two scaling parameters per angle.
2. Per-angle scaling is collapsed onto a small, shared set of
   parameters via the xpcsjax per-angle reparameterisation layer,
   making the multi-angle problem identifiable.
3. :func:`xpcsjax.optimization.nlsq.fit_nlsq` returns a single
   :class:`xpcsjax.optimization.nlsq.results.OptimizationResult` (as for
   homodyne); the joint fit's per-angle detail (``chi2_per_angle``,
   ``contrast_per_angle`` / ``offset_per_angle``) is recorded under
   ``result.nlsq_diagnostics``.

The two-function path
---------------------

The public-API surface is identical to homodyne:

.. code-block:: python

   import xpcsjax

   data = xpcsjax.load_xpcs_data("heterodyne_config.yaml")
   results = xpcsjax.fit_nlsq(data, "heterodyne_config.yaml")

   for i, r in enumerate(results):
       phi = data["phi_angles_list"][i]
       print(f"phi={phi:6.1f} deg  reduced chi2={r.reduced_chi_squared:.3f}")

Note that ``results`` is now a list. The list order matches the order
of ``phi_angles_list`` in the loaded data dictionary.

The two-component physics model
-------------------------------

The two-component (``two_component``) and heterodyne
(``heterodyne``) modes share a single physics kernel. There are
fourteen physics parameters in total, conceptually grouped as:

* **Component 1 dynamics.** Three parameters describing the first
  diffusion process (``D0_1``, ``alpha_1``, ``D_offset_1`` or their
  registry equivalents).
* **Component 2 dynamics.** Three parameters describing the second
  process.
* **Coupling and amplitude.** Parameters controlling the relative
  weight of the two components and any cross-coupling required by the
  experimental geometry.
* **Reference-beam terms.** Parameters that arise from the heterodyne
  reference field (zero in the strict homodyne limit).
* **Per-angle scaling.** Two scaling parameters (typically
  ``contrast`` and ``offset``) that, before reparameterisation, would
  exist independently at every phi angle.

The fourteen-count covers everything except the per-angle scaling,
which is handled separately by the per-angle reparameterisation layer.

The authoritative ordering for the fourteen physics parameters lives
in the parameter registry under :mod:`xpcsjax.config.parameter_registry`
and is accessible through
:meth:`xpcsjax.config.ConfigManager.get_active_parameters`. Always derive your
``initial_parameters.values`` length from that list rather than
hard-coding ``14`` or ``16``.

Per-angle reparameterisation for multi-angle fits
-------------------------------------------------

The naive per-angle scaling parameterisation has two free parameters
per phi angle. For a dataset with, say, 36 angles that is 72 extra
free parameters — far more than the data can constrain. xpcsjax
collapses these onto a small, shared set of parameters selected by the
``per_angle_mode`` setting (``constant`` / ``averaged`` / ``individual``,
with ``auto`` resolving to ``averaged`` or ``individual`` by angle count).

This reparameterisation is implemented in
:mod:`xpcsjax.optimization.nlsq.per_angle_mode`
(:class:`~xpcsjax.optimization.nlsq.per_angle_mode.PerAngleScalingPlan`,
resolved by
:func:`~xpcsjax.optimization.nlsq.per_angle_mode.resolve_per_angle_mode`)
and is enabled automatically for multi-angle heterodyne fits when the
configuration permits. The benefits are:

* The effective parameter dimension stays manageable regardless of
  how many phi angles the dataset has.
* The fitted contrast and offset are shared (``averaged``) or fixed
  (``constant``) rather than a swarm of weakly constrained per-angle
  point estimates.
* Convergence is dramatically more robust than in the ``individual``
  parameterisation, which is essentially degenerate for large
  :math:`N_\phi`.

The mode is controlled by the ``per_angle_mode`` field of the
``optimization.nlsq.anti_degeneracy`` block in the YAML, e.g.:

.. code-block:: yaml

   optimization:
     nlsq:
       max_iterations: 2000
       tolerance: 1.0e-8
       anti_degeneracy:
         per_angle_mode: auto

The ``auto`` default is sensible for typical datasets; only override it
(``constant`` / ``averaged`` / ``individual``) if you have a specific
reason — for example, freezing scaling from quantiles (``constant``) or
forcing fully per-angle scaling (``individual``).

Single-angle fits
-----------------

If the dataset contains a single phi angle, the per-angle
reparameterisation collapses to "constant contrast, constant offset",
i.e. the two scaling parameters revert to ordinary unknowns. The fit
proceeds as a sixteen-parameter problem at that one angle, and the
returned list has length one:

.. code-block:: python

   results = xpcsjax.fit_nlsq(single_angle_data, "single_angle_config.yaml")
   assert len(results) == 1
   r = results[0]

Multi-angle fits
----------------

In the more common multi-angle case, the strata are defined by the
phi-angle filter (see :doc:`/user_guide/data_loading`). The optimiser
walks the strata in the order they appear in the data dictionary and
returns a single :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`.
Its ``nlsq_diagnostics`` records each stratum's chi-squared (``chi2_per_angle``)
and per-angle scaling (``contrast_per_angle`` / ``offset_per_angle``); the 14
physics parameters are shared across strata, while the per-angle scaling is
allowed to vary.

A complete heterodyne example
-----------------------------

A starter heterodyne YAML:

.. code-block:: yaml

   analysis_mode: two_component

   experimental_data:
     data_file_name: heterodyne_dataset.h5

   analyzer_parameters:
     temporal:
       dt: 0.05
       start_frame: 0
       end_frame: 1500
     scattering:
       wavevector_q: 0.015
     geometry:
       stator_rotor_gap: 1.0e-3

   initial_parameters:
     values: [
       # Component 1 (3)
       1.0e3, 0.0, 0.0,
       # Component 2 (3)
       1.0e2, 0.0, 0.0,
       # Coupling + amplitudes + reference-beam terms (8)
       0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       # Per-angle scaling (2)
       0.5, 1.0,
     ]

   parameter_bounds:
     # name -> [lower, upper] for each active parameter
     # ...
   optimization:
     nlsq:
       max_iterations: 2000
       tolerance: 1.0e-8

And a driver script:

.. code-block:: python

   import xpcsjax

   data = xpcsjax.load_xpcs_data("heterodyne_config.yaml")
   results = xpcsjax.fit_nlsq(data, "heterodyne_config.yaml")

   converged = [r for r in results if r.convergence_status == "converged"]
   bad      = [r for r in results if r.quality_flag == "bad"]

   print(f"{len(converged)}/{len(results)} strata converged, "
         f"{len(bad)} flagged 'bad'")

   for phi, r in zip(data["phi_angles_list"], results):
       if r.quality_flag != "good":
           print(f"  phi={phi:6.1f}  status={r.convergence_status}  "
                 f"flag={r.quality_flag}  redchi2={r.reduced_chi_squared:.3f}")

When to dig into per-stratum results
------------------------------------

A heterodyne fit is healthy when most strata land at
``quality_flag == "good"`` and ``convergence_status == "converged"``.
Triaging is per-stratum:

* If only a handful of strata are bad, suspect the underlying data
  quality at those phi angles (beam-stop, detector defects).
* If most strata are bad, the configuration itself is likely
  mis-specified — check bounds, initial values, and the active
  parameter list (see :doc:`/user_guide/analysis_modes`).
* If everything converged but the reduced :math:`\chi^2` values are
  uniformly far from unity, the model family is probably wrong for the
  data — consider whether ``two_component`` is the appropriate
  analysis mode in the first place.

See :doc:`/user_guide/interpreting_results` for the full triage
playbook.

Configuration nesting
---------------------

Heterodyne-specific knobs live under ``optimization.nlsq`` in the YAML
to keep the schema unified with homodyne:

.. code-block:: yaml

   optimization:
     nlsq:
       max_iterations: 2000
       tolerance: 1.0e-8
       multistart:
         n_starts: 8
       anti_degeneracy:
         enabled: true
         per_angle_mode: auto

Any keys not understood by the heterodyne adapter are forwarded
verbatim to the underlying NLSQ solver. The strategy router (see
:doc:`/user_guide/nlsq_fitting`) makes its decision on a per-stratum
basis, so a single fit can mix in-memory and stratified-LS strategies
across phi-angle groups.

Where to look next
------------------

* :doc:`/user_guide/nlsq_fitting` — strategy selection, bounds,
  parameter transforms, multistart and CMA-ES escape.
* :doc:`/user_guide/interpreting_results` — meaning of every field
  on each per-stratum result.
