Interpreting results
====================

.. currentmodule:: xpcsjax


:func:`xpcsjax.optimization.nlsq.fit_nlsq` returns a single
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult` for every analysis
mode (homodyne and heterodyne alike); for heterodyne the joint multi-angle fit
records its per-angle detail under ``nlsq_diagnostics``. This page documents that
result shape and explains what each field is useful for.

The result dataclass
--------------------

:class:`xpcsjax.optimization.nlsq.results.OptimizationResult` exposes:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Field
     - Type
     - Meaning
   * - ``parameters``
     - 1-D array
     - Fitted values for the active parameters, in registry order.
   * - ``uncertainties``
     - 1-D array
     - 1-:math:`\sigma` standard errors aligned with ``parameters``.
   * - ``covariance``
     - 2-D array
     - Full parameter covariance matrix; symmetric; positive
       semi-definite at convergence.
   * - ``chi_squared``
     - float
     - Total :math:`\chi^2` at the optimum.
   * - ``reduced_chi_squared``
     - float
     - :math:`\chi^2 / (N - p)` where :math:`N` is the number of
       residuals and :math:`p` the active parameter count.
   * - ``convergence_status``
     - str
     - One of ``"converged"``, ``"max_iter"``, ``"failed"``, ``"partial"``.
   * - ``iterations``
     - int
     - Number of trust-region iterations consumed.
   * - ``execution_time``
     - float
     - Wall-clock seconds for the fit (excluding compile).
   * - ``device_info``
     - dict
     - JAX device record (CPU only in v0.1).
   * - ``recovery_actions``
     - list[str]
     - Trail of interventions by the anti-degeneracy controller and
       fallback chain, as short string tags.
   * - ``quality_flag``
     - str
     - One of ``"good"``, ``"marginal"``, ``"poor"``, ``"unknown"``. See triage below.
   * - ``streaming_diagnostics``
     - dict
     - Strategy router decisions, hybrid-streaming chunking, memory
       accounting.
   * - ``stratification_diagnostics``
     - dict
     - Per-stratum chunk sizes, per-angle reparameterisation mode,
       dropped angles.
   * - ``nlsq_diagnostics``
     - dict
     - Residual history, Jacobian conditioning, trust-region size.
   * - ``sigma_is_default``
     - bool
     - ``True`` if uncertainties were computed from a default
       :math:`\sigma`; ``False`` if data-driven.

Two convenience properties are exposed:

:attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.success`
    ``True`` iff ``convergence_status == "converged"``. It does not
    consult ``quality_flag`` -- a converged fit can still carry a
    ``"poor"`` quality flag, so inspect both.

:attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.message`
    A short human-readable summary suitable for logging or display.

Parameter ordering
------------------

The order of entries in ``parameters``, ``uncertainties``, and the
rows/columns of ``covariance`` is the order returned by
:meth:`xpcsjax.config.ConfigManager.get_active_parameters`. Do not hard-code the
ordering — always derive it from the configuration:

.. code-block:: python

   from xpcsjax import ConfigManager
   cfg = ConfigManager("config.yaml")
   cfg.load_config()

   names = cfg.get_active_parameters()
   for name, value, sigma in zip(names, result.parameters, result.uncertainties):
       print(f"{name:>12s} = {value:.4e} ± {sigma:.2e}")

For ``static`` and ``static_isotropic`` modes, the order is
``["D0", "alpha", "D_offset"]``. For ``laminar_flow`` it extends to
seven entries. For ``two_component`` / ``heterodyne`` it is the
fourteen physics parameters returned by the registry, with per-angle
scaling handled separately by the per-angle reparameterisation layer.

Reduced chi-squared interpretation
----------------------------------

The reduced statistic is:

.. math::

   \chi^2_\nu \;=\; \frac{\chi^2}{N - p}

where :math:`N` is the number of independent residuals included in
the fit and :math:`p` the active parameter count.

Reading the value:

* :math:`\chi^2_\nu \approx 1` — the model fits the data to within the
  estimated noise. Default success criterion.
* :math:`\chi^2_\nu \gg 1` — the model is failing to explain features
  in the data. Could be a wrong analysis mode, a missing physics
  term, or systematic features at angles not handled by the chosen
  parameterisation.
* :math:`\chi^2_\nu \ll 1` — the uncertainties used in the residual
  weighting are overestimated, or the model is over-parameterised for
  the data.

If :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.sigma_is_default` is ``True``, the :math:`\sigma` used in
the residual weighting was a default value and the absolute scale of
:math:`\chi^2_\nu` should be treated as advisory rather than
absolute. The ranking between fits is still meaningful.

Quality flag triage
-------------------

The ``quality_flag`` is a coarse four-level summary driven by the
reduced :math:`\chi^2` band and the number of parameters pinned at a
bound:

``"good"``
    Reduced :math:`\chi^2 < 2.0` and no parameter ended up at a bound.

``"marginal"``
    Reduced :math:`\chi^2 < 5.0` with at most two parameters at a bound
    — converged, but worth a second look at the diagnostics.

``"poor"``
    Reduced :math:`\chi^2 \ge 5.0`, several parameters pinned at bounds,
    or the solve failed / was forced. Do not use the parameter values
    for scientific reporting without re-inspecting the diagnostics.

``"unknown"``
    The reduced :math:`\chi^2` was ``None`` or non-finite, so the fit
    could not be classified.

Always inspect ``recovery_actions`` when ``quality_flag`` is
``"marginal"``, ``"poor"``, or ``"unknown"``. The structured trail
tells you which intervention fired and what its outcome was.

Reading ``recovery_actions``
----------------------------

Each entry is a short ``str`` tag naming one intervention:

.. code-block:: python

   for action in result.recovery_actions:
       print(action)

Typical tags are:

* ``"detected_parameter_stagnation"`` — the controller saw the solver
  stall.
* ``"stagnation_after_all_retries"`` — retries did not break the stall.
* ``"strategy_fallback_to_<strategy>"`` — the strategy router fell back
  to another strategy mid-fit (e.g. ``strategy_fallback_to_out_of_core``).

The trail is a plain ``list[str]``, append-only, and survives
serialisation, so it can be audited after the fact.

The diagnostics dictionaries
----------------------------

Three structured diagnostic blocks are attached to every result.

``streaming_diagnostics``
~~~~~~~~~~~~~~~~~~~~~~~~~

Routing-time decisions:

* Strategy tag chosen by ``select_nlsq_strategy``.
* RAM budget at decision time vs. estimated working-set size.
* Chunking layout for hybrid-streaming and out-of-core strategies.

This block is the first place to look when you suspect the wrong
strategy was selected (for example, an in-memory fit that is much
slower than expected; or a hybrid-streaming fit on a small dataset).

``stratification_diagnostics``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-stratum bookkeeping:

* Number of phi-angle strata.
* Effective parameter dimension after per-angle reparameterisation.
* Any phi angles dropped by the filter.

For heterodyne fits this block is essentially mandatory reading.

``nlsq_diagnostics``
~~~~~~~~~~~~~~~~~~~~

Solver-level traces:

* Residual norm history per iteration.
* Trust-region size history.
* Final Jacobian conditioning.
* Termination reason as reported by the underlying NLSQ solver.

A monotonically decreasing residual that ran out of iterations is a
strong signal to raise ``max_iterations``. A residual that plateaus
high is a strong signal to check bounds and/or the analysis mode.

Per-angle results in heterodyne fits
------------------------------------

For ``two_component`` and ``heterodyne`` modes, :func:`xpcsjax.fit_nlsq`
still returns a **single** :class:`OptimizationResult` -- the joint fit
across all phi-angles. The per-angle detail is recorded under
``result.nlsq_diagnostics`` (e.g. ``chi2_per_angle``, the per-angle
scaling, and ``per_angle_mode``). Read the per-angle reduced
:math:`\chi^2` with the helper in
:mod:`xpcsjax.optimization.nlsq.heterodyne_views`:

.. code-block:: python

   from xpcsjax.optimization.nlsq.heterodyne_views import per_angle_chi2

   result = xpcsjax.fit_nlsq(data, "heterodyne_config.yaml")
   chi2 = per_angle_chi2(result)          # one entry per phi stratum
   for phi, c in zip(data["phi_angles_list"], chi2):
       print(f"  phi={phi:6.1f}  reduced_chi2={c:.3g}")

The headline parameters and uncertainties on ``result`` already describe
the joint fit; ``per_angle_chi2`` is the diagnostic for spotting which
strata fit worst.

Serialisation
-------------

The result object is a plain dataclass: every field is a JSON-friendly
scalar, array, or nested ``dict`` of the same. The convenience
``save`` helpers documented in the module docstring write the
parameters, uncertainties, and diagnostics to a structured directory
of JSON and NumPy ``.npy`` files suitable for downstream analysis and
plotting. Avoid serialising with arbitrary-object formats; the JSON +
``.npy`` layout is the supported on-disk contract.

What to read next
-----------------

* :doc:`/user_guide/troubleshooting` — what to do when the result
  comes back with ``quality_flag`` of ``"marginal"`` or ``"poor"``.
* :doc:`/user_guide/nlsq_fitting` — the mechanics behind the
  diagnostics on the result.
