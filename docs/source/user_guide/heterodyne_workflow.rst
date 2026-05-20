Heterodyne workflow
===================

.. currentmodule:: xpcsjax


The heterodyne pipeline fits the two-component XPCS model
(``analysis_mode: two_component`` or its alias ``"heterodyne"``)
against multi-angle two-time correlation data. It differs from the
homodyne pipeline in three places:

1. The parameter set is much larger — fourteen physics parameters
   plus two scaling parameters per angle.
2. Per-angle scaling is collapsed onto a small number of Fourier
   coefficients via the xpcsjax Fourier reparameterisation layer,
   making the multi-angle problem identifiable.
3. The return value of :func:`xpcsjax.optimization.nlsq.fit_nlsq` is a
   ``list[NLSQResult]`` (one entry per phi-angle group) rather than a
   single :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`.

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
which is handled separately by the Fourier reparameterisation layer.

The authoritative ordering for the fourteen physics parameters lives
in the parameter registry under :mod:`xpcsjax.config.parameter_registry`
and is accessible through
:meth:`xpcsjax.config.ConfigManager.get_active_parameters`. Always derive your
``initial_parameters.values`` length from that list rather than
hard-coding ``14`` or ``16``.

Fourier reparameterisation for multi-angle fits
-----------------------------------------------

The naive per-angle scaling parameterisation has two free parameters
per phi angle. For a dataset with, say, 36 angles that is 72 extra
free parameters — far more than the data can constrain. xpcsjax
collapses these onto a smooth angular function expressed as a small
truncated Fourier series.

This reparameterisation is implemented in
:mod:`xpcsjax.optimization.nlsq.fourier_reparam` and is enabled
automatically for multi-angle heterodyne fits when the configuration
permits. The benefits are:

* The effective parameter dimension stays manageable regardless of
  how many phi angles the dataset has.
* The fitted contrast and offset functions are guaranteed to be
  smooth functions of phi rather than independent point estimates.
* Convergence is dramatically more robust than in the per-angle
  parameterisation, which is essentially degenerate.

The order of the Fourier expansion is controlled by the
``optimization.nlsq.fourier`` block in the YAML, e.g.:

.. code-block:: yaml

   optimization:
     nlsq:
       max_iterations: 2000
       tolerance: 1.0e-8
       fourier:
         contrast_order: 2
         offset_order: 1

Defaults are sensible for typical datasets; only override them if you
have a specific reason (for example, known higher-order angular
modulation in the contrast).

Single-angle fits
-----------------

If the dataset contains a single phi angle, the Fourier
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
returns one ``NLSQResult`` per stratum. Each result records its own
parameters, uncertainties, covariance, and diagnostics — they are not
shared across strata even though the Fourier coefficients tying them
together are shared during optimisation.

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
       fourier:
         contrast_order: 2
         offset_order: 1
       multistart:
         n_starts: 8
       anti_degeneracy:
         enabled: true

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
