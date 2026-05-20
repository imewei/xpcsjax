NLSQ fitting
============

.. currentmodule:: xpcsjax


This page covers what happens inside :func:`xpcsjax.optimization.nlsq.fit_nlsq` between
the moment you pass it a data dictionary and the moment it returns an
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult`. The mechanics matter because, in
practice, "the fit didn't converge" is the most common operational
issue with XPCS data, and the path to fixing it goes through the
strategy router, the bounds layer, and (if necessary) the
anti-degeneracy controller.

Division of labour with the upstream NLSQ library
-------------------------------------------------

xpcsjax depends on the upstream ``nlsq>=0.6.10`` package. The split of
responsibilities is non-negotiable:

NLSQ owns
~~~~~~~~~

* The ``CurveFit`` JIT cache.
* The ``curve_fit`` entry point and its trust-region (Levenberg–
  Marquardt) inner solver.
* All low-level Jacobian and Gauss–Newton step machinery.

xpcsjax owns
~~~~~~~~~~~~

* Strategy selection (``select_nlsq_strategy``): in-memory vs
  stratified-LS vs hybrid-streaming vs out-of-core.
* The five-layer anti-degeneracy controller
  (:mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`).
* CMA-ES escape, auto-triggered above a configurable failure threshold
  (:mod:`xpcsjax.optimization.nlsq.cmaes_wrapper`).
* Latin Hypercube Sampling (LHS) multistart
  (:mod:`xpcsjax.optimization.nlsq.multistart`).
* Bounds plumbing and parameter transforms
  (:mod:`xpcsjax.optimization.nlsq.transforms`).
* Angle-stratified chunking for large datasets.
* Shear-weighting for laminar-flow analyses
  (:mod:`xpcsjax.optimization.nlsq.shear_weighting`).

.. warning::

   Do not call ``nlsq.fit()`` (NLSQ's higher-level unified entry point)
   from within xpcsjax. xpcsjax routes memory and strategy itself.
   ``CurveFit`` is the only NLSQ entry point we use.

   The previous-generation ``WorkflowSelector`` was removed in NLSQ
   v0.6.0; if you see code calling it anywhere in xpcsjax, that is a
   regression to be removed, not an extension point.

Strategy selection
------------------

``select_nlsq_strategy`` is the first thing :func:`xpcsjax.optimization.nlsq.fit_nlsq` runs once
the configuration is parsed. It inspects:

* The shape and dtype of ``c2_exp``.
* The number of phi-angle strata.
* The active parameter count for the configured analysis mode.
* The current process memory budget.

…and returns one of four strategy tags:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Strategy
     - When it is selected
   * - in-memory
     - Datasets that fit comfortably in RAM with headroom for the
       Jacobian. The default for typical static and isotropic fits.
   * - stratified-LS
     - Many phi-angle strata that individually fit in memory but whose
       joint Jacobian would not. Each stratum is solved independently
       with shared parameters via the Fourier reparameterisation
       layer.
   * - hybrid-streaming
     - Very large datasets (typically 10M+ correlation entries). Data
       arrays are captured in JIT closures; this is the case where the
       package-level ``XLA_FLAGS`` ``constant_folding`` disable is load
       bearing for compile times.
   * - out-of-core
     - Datasets too large for any in-memory strategy. The residual is
       accumulated over disk-backed chunks.

The selected strategy is recorded on the returned
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult` in ``streaming_diagnostics`` and
``stratification_diagnostics``, so you can always see which path the
fit took.

Bounds and parameter transforms
-------------------------------

Bounds are specified in YAML under ``parameter_bounds`` as
``[lower, upper]`` pairs per parameter name. They are pulled by
:meth:`xpcsjax.config.ConfigManager.get_parameter_bounds` and threaded into the
optimiser through the xpcsjax parameter-transform layer.

The transform layer maps each bounded parameter through a smooth
bijection onto an unconstrained real line, lets NLSQ's trust-region
solver run on the unconstrained problem, and inverts the transform
when materialising parameters. This is conceptually different from a
naive clipping approach — gradients are well-defined at the bound
edges and the Jacobian remains finite.

Practical consequences:

* Bounds are always honoured. There is no path through the optimiser
  that violates them.
* If you set a bound very close to a fitted parameter, the
  uncertainty estimate inherits the transform's curvature near the
  bound. Wide, physically-motivated bounds are preferable to tight
  ones.
* For diffusion coefficients spanning many orders of magnitude, the
  transform is the principal reason that ``JAX_ENABLE_X64=1`` is
  mandatory at the package level.

Bound configuration accepts both per-parameter dict form and
mode-specific list form. The dict form is checked against the active
parameter list at load time:

.. code-block:: yaml

   parameter_bounds:
     D0:       [1.0e1, 1.0e5]
     alpha:    [-1.0, 1.0]
     D_offset: [-1.0e3, 1.0e3]

Convergence criteria
--------------------

Convergence is governed by:

* ``optimization.nlsq.tolerance`` — relative residual change between
  iterations.
* ``optimization.nlsq.max_iterations`` — hard cap on iteration count.
* The internal trust-region step-size floor.

A run terminates with one of three statuses, recorded on the result
as :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.convergence_status`:

``"converged"``
    Tolerance reached within the iteration budget. Most fits should
    land here.

``"max_iter"``
    Iteration cap hit without satisfying the tolerance. This is
    typically a sign that the initial point was very far from the
    solution, or that bounds are too tight, or that the model family
    is mis-specified.

``"failed"``
    The solver could not produce a valid step (numerical breakdown,
    NaN/Inf in the Jacobian, etc.). When this status appears, check
    the ``recovery_actions`` trail on the result; xpcsjax usually
    attempts at least one rescue before giving up.

What to do when convergence_status != 'converged'
-------------------------------------------------

1. Check :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.recovery_actions`. It is a structured trail of every
   intervention the controller applied (parameter clipping, transform
   adjustments, multistart, CMA-ES escape). If an intervention failed,
   the entry records why.
2. Inspect ``nlsq_diagnostics`` for the residual history. A
   monotonically decreasing residual that ran out of iterations
   usually means raising ``max_iterations``; a residual that plateaus
   well above what your data quality justifies usually means the bound
   on one of the parameters is too tight.
3. Re-run with multistart enabled
   (``optimization.nlsq.multistart.n_starts: 8`` is a reasonable
   default). The LHS multistart fights initial-condition sensitivity,
   which is by far the most common cause of ``max_iter``.
4. If multistart cannot rescue the fit, enable the CMA-ES escape. This
   is auto-triggered above a configurable failure threshold, but you
   can also force it via the configuration.

Multistart
----------

The multistart driver (:mod:`xpcsjax.optimization.nlsq.multistart`)
samples ``n_starts`` initial points from a Latin Hypercube over the
bounded parameter region, runs an independent NLSQ fit from each, and
returns the best by reduced :math:`\chi^2`. The cost is linear in
``n_starts``; for any non-trivial XPCS analysis this is essentially
always worth running.

The multistart is per-fit, not per-stratum: in a heterodyne run with
36 strata, multistart applies to the global Fourier-coefficient
parameters, with each stratum still solved deterministically given
those globals.

CMA-ES escape
-------------

When NLSQ multistart fails repeatedly, xpcsjax falls back to a
Covariance Matrix Adaptation Evolution Strategy
(:mod:`xpcsjax.optimization.nlsq.cmaes_wrapper`). CMA-ES is a
derivative-free global optimiser; it is slower than NLSQ by orders of
magnitude but is robust against landscapes that defeat trust-region
methods (very narrow valleys, near-degenerate parameter pairs, etc.).

CMA-ES escape is recorded on the result as a ``recovery_action``
entry. The downstream parameter and uncertainty estimates come from a
final NLSQ polish step seeded by the CMA-ES point — so the returned
covariance is still derived from a local Jacobian, not from the CMA-ES
ensemble.

Anti-degeneracy controller
--------------------------

The anti-degeneracy controller is the five-layer system in
:mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`. It is on
by default. Its layers, in increasing severity:

1. **Numerical hygiene.** Replace non-finite Jacobian rows; clip
   parameters back into bounds.
2. **Adaptive regularisation.** Add a Levenberg damping term scaled
   by the residual magnitude.
3. **Gradient monitoring.** Detect saddle-like behaviour and inject a
   small perturbation.
4. **Multistart re-launch.** Re-seed from a fresh LHS point if the
   solver is stuck.
5. **CMA-ES escape.** As above.

The controller is conservative — it intervenes only when a degeneracy
is detected, not preemptively. Every intervention shows up in
``recovery_actions``, so you can audit it after the fact.

What goes on the result
-----------------------

Every action taken by the routing, multistart, escape, and
anti-degeneracy systems is reflected in
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult`:

* :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.streaming_diagnostics` — strategy router decisions and
  hybrid-streaming chunk accounting.
* :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.stratification_diagnostics` — per-stratum chunk sizes,
  Fourier reparameterisation degrees, dropped angles.
* :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.nlsq_diagnostics` — residual history, Jacobian conditioning,
  trust-region size.
* :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.recovery_actions` — controller interventions, multistart
  draws, CMA-ES escape provenance.

Read these fields in concert when triaging a fit. See
:doc:`/user_guide/interpreting_results`.

Next: :doc:`/user_guide/interpreting_results` documents every field
on the result object in detail.
