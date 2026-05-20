Interpreting results
====================

.. currentmodule:: xpcsjax


:func:`xpcsjax.optimization.nlsq.fit_nlsq` returns either a single
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult` (homodyne) or a ``list[NLSQResult]``
(heterodyne / two-component). The two types share their public field
shape; this page documents that shape and explains what each field is
useful for.

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
     - One of ``"converged"``, ``"max_iter"``, ``"failed"``.
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
     - list[dict]
     - Trail of interventions by the anti-degeneracy controller and
       fallback chain.
   * - ``quality_flag``
     - str
     - One of ``"good"``, ``"warn"``, ``"bad"``. See triage below.
   * - ``streaming_diagnostics``
     - dict
     - Strategy router decisions, hybrid-streaming chunking, memory
       accounting.
   * - ``stratification_diagnostics``
     - dict
     - Per-stratum chunk sizes, Fourier reparameterisation degrees,
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
    ``True`` iff ``convergence_status == "converged"`` and
    ``quality_flag != "bad"``.

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
scaling handled separately by the Fourier layer.

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

The ``quality_flag`` is a coarse three-level summary:

``"good"``
    Convergence is clean, residuals look sensible, no recovery actions
    were needed.

``"warn"``
    Fit converged but the controller flagged at least one warning sign
    — for example, the parameter ended up at a bound, a recovery
    action was applied, or the reduced :math:`\chi^2` is outside the
    expected band.

``"bad"``
    The fit either failed (``convergence_status`` is ``"failed"`` or
    ``"max_iter"``) or recovered only via aggressive CMA-ES fallback.
    Do not use the parameter values for scientific reporting without
    re-inspecting the diagnostics.

Always inspect ``recovery_actions`` when ``quality_flag`` is
``"warn"`` or ``"bad"``. The structured trail tells you which
intervention fired and what its outcome was.

Reading ``recovery_actions``
----------------------------

Each entry is a ``dict`` describing one intervention:

.. code-block:: python

   for action in result.recovery_actions:
       print(action["stage"], action["action"], action.get("outcome"))

Typical stages are:

* ``"strategy_router"`` — strategy was changed mid-fit.
* ``"anti_degeneracy"`` — one of the five anti-degeneracy layers
  triggered.
* ``"multistart"`` — a multistart re-seed produced a better point.
* ``"cmaes_escape"`` — CMA-ES global escape was invoked.
* ``"polish"`` — final NLSQ polish step after a CMA-ES escape.

The trail is append-only and survives serialisation, so it can be
audited after the fact.

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
* Effective parameter dimension after Fourier reparameterisation.
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

For ``two_component`` and ``heterodyne`` modes, the return value is a
list with one entry per phi-angle stratum. Iterate the list against
``data["phi_angles_list"]``:

.. code-block:: python

   results = xpcsjax.fit_nlsq(data, "heterodyne_config.yaml")
   for phi, r in zip(data["phi_angles_list"], results):
       if r.quality_flag != "good":
           print(f"  phi={phi:6.1f}  status={r.convergence_status}  "
                 f"flag={r.quality_flag}")

Aggregating across strata is dataset-specific. A common pattern is to
compute the mean and dispersion of each physics parameter across the
"good" strata and to use that as the headline reported value, with the
"warn" and "bad" strata recorded but excluded:

.. code-block:: python

   good = [r for r in results if r.quality_flag == "good"]
   import numpy as np
   params = np.stack([r.parameters for r in good], axis=0)
   mean   = params.mean(axis=0)
   std    = params.std(axis=0)

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
  comes back with ``quality_flag`` of ``"warn"`` or ``"bad"``.
* :doc:`/user_guide/nlsq_fitting` — the mechanics behind the
  diagnostics on the result.
