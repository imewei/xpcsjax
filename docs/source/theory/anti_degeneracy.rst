.. _theory_anti_degeneracy:

Anti-Degeneracy Defence
=======================

Fitting the homodyne laminar-flow or the heterodyne two-component kernel
simultaneously across :math:`N_\phi` azimuthal angles is mathematically
ill-conditioned without additional structure. The per-angle scaling
parameters :math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))` are degenerate
with the physical parameters --- principally :math:`D_0` and
:math:`\dot{\gamma}_0` --- and the shear gradient cancels when summed over
angles, producing a flat optimisation landscape that collapses
:math:`\dot{\gamma}_0` to a non-physical value. The xpcsjax
:mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller` orchestrates a
**five-layer defence** that breaks the degeneracy, addresses the gradient
cancellation, and monitors the optimisation in real time.

This page explains the degeneracy mechanism, walks through each of the five
layers, and points to the implementing module. The implementation closely
follows the strategy first introduced in the upstream homodyne package and
ported into xpcsjax for v0.1.

The parameter absorption degeneracy
-----------------------------------

At a single angle :math:`\phi_k` the laminar-flow kernel
(:eq:`hm_c2_laminar`) is

.. math::
   :label: ad_c2_single

   c_2(\phi_k, t_1, t_2; \theta)
   \;=\; c_\mathrm{offset}(\phi_k)
   \;+\; \beta(\phi_k)\,
       \exp\!\left(-q^2\!\int_{t_1}^{t_2} J(t')\,dt'\right)
       \mathrm{sinc}^2\!\left(\tfrac{q h \cos(\phi - \phi_0)\,\Gamma(t_1, t_2)}{2\pi}\right).

If :math:`\beta(\phi_k)` and :math:`c_\mathrm{offset}(\phi_k)` are treated as
independent free parameters per angle, the optimisation landscape has a
**flat direction**:

   Increasing :math:`D_0 \to D_0 + \delta` and simultaneously rescaling
   :math:`\beta(\phi_k) \to \beta(\phi_k)\, e^{q^2 \delta\, t_\mathrm{ref}}`
   produces identical :math:`c_2` values across all angles. The physical
   parameters are **not identifiable** from the per-angle contrasts without
   a constraint.

This degeneracy is generic whenever:

1. Per-angle :math:`\beta(\phi_k)` and :math:`c_\mathrm{offset}(\phi_k)` are
   freely optimised;
2. The number of angle-specific parameters exceeds the information content
   per angle;
3. The diffusion contribution and the contrast contribution share the same
   functional form.

The gradient cancellation problem
---------------------------------

The shear term in :math:`c_2` introduces an angle-dependent piece whose
gradient with respect to :math:`\dot{\gamma}_0` is proportional to
:math:`\cos(\phi - \phi_0)`. Summed over angles that span :math:`[0, 2\pi)`,
positive and negative contributions partially cancel:

.. code-block:: text

   Example for 8 equally spaced angles, phi_0 = 0:

       phi = 0   :  cos = +1.00 ----+
       phi = 45  :  cos = +0.71     | partially cancel when summed
       phi = 90  :  cos =  0.00     |
       phi = 135 :  cos = -0.71     |
       phi = 180 :  cos = -1.00 ----+
       ...

The net gradient on :math:`\dot{\gamma}_0` is weak, and the optimiser
finds it easier to absorb the angle dependence into per-angle
:math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))` than to drive
:math:`\dot{\gamma}_0` toward its true value. The result is parameter
collapse: :math:`\dot{\gamma}_0` floats to its lower bound.

The five-layer defence
----------------------

The :class:`~xpcsjax.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController`
orchestrates five complementary mechanisms. The layers are not redundant:
each addresses a different root cause and they compose.

Layer 1 -- Fourier / constant reparameterisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module:** :mod:`xpcsjax.optimization.nlsq.fourier_reparam`.
**Class:** :class:`~xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer`.

This layer attacks the structural degeneracy by reducing the dimension of
the per-angle scaling space. Four modes are available, selected through the
``per_angle_mode`` setting:

* ``constant`` -- per-angle :math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))`
  are estimated from data quantiles and held **fixed** during the fit. Only
  the physical parameters are optimised. Total parameters: physical only.
* ``auto`` (default) -- for :math:`N_\phi \geq 3`, computes the quantile
  estimates, averages them to a single
  :math:`(\bar{\beta}, \bar{c}_\mathrm{offset})`, and optimises these two
  averaged scalars together with the physical parameters. For
  :math:`N_\phi < 3`, falls back to ``individual``.
* ``fourier`` -- expresses :math:`\beta(\phi)` and
  :math:`c_\mathrm{offset}(\phi)` as a truncated Fourier series of order
  :math:`K` (default :math:`K = 2`),

  .. math::
     :label: ad_fourier

     \beta(\phi)
     \;=\; \bar{\beta}
     + \sum_{\ell=1}^K \bigl[a_\ell \cos(\ell \phi) + b_\ell \sin(\ell \phi)\bigr],

  yielding :math:`2 K + 1 = 5` coefficients per group instead of
  :math:`N_\phi` per group.
* ``individual`` -- each angle has independent
  :math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))`, adding
  :math:`2 N_\phi` free parameters. Use only for very small :math:`N_\phi`
  or as a post-hoc refinement of an ``auto``-mode fit.

The quantile estimation underlying ``constant`` and ``auto`` exploits the
Siegert plateau:

* At small lags (\ :math:`\Delta t \to 0`),
  :math:`c_2 \to \beta + c_\mathrm{offset}` (the ceiling).
* At large lags (\ :math:`\Delta t \to \infty`), :math:`c_2 \to c_\mathrm{offset}`
  (the floor).

The 90th percentile of small-lag values gives a robust ceiling, the 10th
percentile of large-lag values gives a robust floor, and
:math:`\beta = \text{ceiling} - \text{floor}` follows. Quantiles are used
instead of min / max for outlier robustness.

**Parameter count for a 23-angle laminar-flow fit**:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Mode
     - Parameters
     - Notes
   * - ``constant``
     - 7
     - Scaling fixed from quantiles; fastest convergence.
   * - ``auto``
     - 9
     - 7 physical + 2 averaged scaling; recommended default.
   * - ``fourier`` (K=2)
     - 17
     - 7 physical + 10 Fourier coefficients.
   * - ``individual``
     - 53
     - 7 physical + 46 per-angle; high degeneracy risk.

Layer 2 -- Hierarchical two-stage optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module:** :mod:`xpcsjax.optimization.nlsq.hierarchical`.
**Class:** :class:`~xpcsjax.optimization.nlsq.hierarchical.HierarchicalOptimizer`.

This layer breaks gradient cancellation by alternating between two
optimisation stages that operate on disjoint parameter blocks.

**Stage 1 --- physical parameters only.** Per-angle scaling parameters are
frozen at their current values. The trust-region solver receives the full
gradient signal on :math:`(D_0, \alpha, D_\mathrm{offset},
\dot{\gamma}_0, \beta_\gamma, \dot{\gamma}_\mathrm{offset}, \phi_0)`
without dilution from the scaling block.

**Stage 2 --- per-angle parameters only.** Physical parameters are frozen at
the Stage 1 result. The per-angle parameters adjust to match the fixed
physics model.

The two stages alternate until the change in the physical parameter block
falls below the outer tolerance or the maximum outer iteration count is
reached. The alternation prevents either block from absorbing signal that
properly belongs to the other.

Layer 3 -- Adaptive CV-based regularisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module:** :mod:`xpcsjax.optimization.nlsq.adaptive_regularization`.
**Class:** :class:`~xpcsjax.optimization.nlsq.adaptive_regularization.AdaptiveRegularizer`.

Classical variance-penalty regularisation
:math:`L_\mathrm{reg} = \lambda\,\mathrm{Var}(\text{params})\cdot N` is
typically swamped by the data loss and contributes a negligible fraction
(\ :math:`\sim 0.01\%`) of the total objective. Layer 3 replaces it with a
relative penalty based on the coefficient of variation,

.. math::
   :label: ad_cv

   \mathrm{CV} \;=\; \frac{\mathrm{std}(\text{params})}{|\mathrm{mean}(\text{params})|},
   \qquad
   L_\mathrm{reg} \;=\; \lambda \cdot \mathrm{CV}^2 \cdot \mathrm{MSE} \cdot N.

With :math:`\lambda` auto-tuned so that the penalty contributes a target
fraction (typically :math:`10\%`) of MSE at a target CV (typically
:math:`0.10`), the regularisation becomes scale-invariant, physically
interpretable, and large enough to actually constrain the optimisation.

Layer 4 -- Gradient collapse monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module:** :mod:`xpcsjax.optimization.nlsq.gradient_monitor`.
**Class:** :class:`~xpcsjax.optimization.nlsq.gradient_monitor.GradientCollapseMonitor`.

The fourth layer monitors the optimisation in real time and detects
**gradient collapse** --- the state in which physical-parameter gradients
become negligible compared to per-angle gradients. The detection criterion
is

.. math::
   :label: ad_ratio

   \mathrm{ratio}
   \;=\; \frac{\|\nabla_\mathrm{physical} L\|}{\|\nabla_\mathrm{per\text{-}angle} L\|}.

When :math:`\mathrm{ratio} < \tau` (default :math:`10^{-2}`) for
:math:`N_c` consecutive iterations (default :math:`5`), collapse is
declared and one of four responses is taken:

* ``warn`` -- log a warning and continue;
* ``hierarchical`` -- switch to Layer 2 alternation;
* ``reset`` -- reset per-angle parameters to their mean values;
* ``abort`` -- terminate with a diagnostic.

The default response is ``hierarchical``, which composes naturally with
Layer 2.

Layer 5 -- Shear-sensitivity weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module:** :mod:`xpcsjax.optimization.nlsq.shear_weighting`.
**Class:** :class:`~xpcsjax.optimization.nlsq.shear_weighting.ShearSensitivityWeighting`.

The fifth layer addresses gradient cancellation directly by weighting
residuals according to their sensitivity to the shear parameter. The
shear-term gradient at angle :math:`\phi` scales with
:math:`|\cos(\phi - \phi_0)|`, so the weight assigned to angle :math:`\phi`
is

.. math::
   :label: ad_weight

   w(\phi)
   \;=\; w_\mathrm{min}
   + (1 - w_\mathrm{min})\,
     \bigl|\cos(\phi - \phi_0)\bigr|^{\,a},

with defaults :math:`w_\mathrm{min} = 0.3` and exponent :math:`a = 1.0`.
The weights are normalised so that their mean equals one, preserving the
loss scale.

The effect is to amplify residuals at shear-sensitive angles
(\ :math:`\phi \approx \phi_0` or :math:`\phi \approx \phi_0 + \pi`) and
attenuate residuals at shear-insensitive angles
(\ :math:`\phi \approx \phi_0 \pm \pi/2`). The asymmetric weighting breaks
the gradient cancellation symmetry and produces a net signal on
:math:`\dot{\gamma}_0`.

.. note::

   Layer 5 is **homodyne-specific**: it only activates for the
   ``static``, ``static_isotropic``, and ``laminar_flow`` modes, where a
   shear rate appears in the kernel. For the heterodyne ``two_component``
   mode the layer short-circuits because there is no shear-rate parameter
   to weight; the analogous physics is carried by the velocity-encoding
   cosine in :eq:`het_Anm`. The layer gating is declared in
   ``_LAYER_GATES`` inside
   :mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`.

Layer-by-layer coverage by optimisation path
--------------------------------------------

The five layers compose differently depending on the optimisation path.

.. list-table:: Layer coverage by optimisation path
   :header-rows: 1
   :widths: 32 14 14 14 14 14

   * - Path
     - Layer 1
     - Layer 2
     - Layer 3
     - Layer 4
     - Layer 5
   * - Local NLSQ (gradient)
     - yes
     - yes
     - yes
     - yes
     - yes
   * - Multistart (LHS)
     - yes
     - yes
     - yes
     - yes
     - yes
   * - CMA-ES escape
     - yes
     - --
     - --
     - --
     - --

Layers 2--5 are specific to gradient-based optimisation; CMA-ES uses
fitness ranking rather than gradients, so the hierarchical alternation,
the gradient monitor, and the gradient-cancellation weighting do not
apply. Layer 1 (parameter-space reduction) is, however, essential for
CMA-ES too --- it reduces the search dimension from :math:`53` to
:math:`9` for a 23-angle laminar-flow fit and is the difference between a
tractable and an intractable global search.

Configuration
-------------

The controller is configured by an
:class:`~xpcsjax.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig`
dataclass with the following key fields:

* ``enable`` (default ``True``) -- master switch;
* ``per_angle_mode`` (default ``"auto"``) -- one of
  ``"individual"``, ``"constant"``, ``"fourier"``, ``"auto"``;
* ``fourier_order`` (default 2) -- truncation order :math:`K`;
* ``constant_scaling_threshold`` (default 3) -- :math:`N_\phi` at which
  ``"auto"`` switches from ``"individual"`` to ``"constant"``;
* ``hierarchical_enable`` (default ``True``) -- Layer 2 on/off;
* ``hierarchical_max_outer_iterations`` (default 5) -- maximum outer loops;
* ``regularization_mode`` (default ``"relative"``) -- one of
  ``"absolute"``, ``"relative"``, ``"auto"``;
* ``gradient_monitoring_enable`` (default ``True``) -- Layer 4 on/off;
* ``gradient_ratio_threshold`` (default :math:`10^{-2}`) -- collapse
  threshold;
* ``gradient_response_mode`` (default ``"hierarchical"``) -- response on
  collapse.

The full set of fields is enumerated in
:mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`. Layer 5's
parameters (\ :math:`w_\mathrm{min}`, :math:`a`, update frequency) live on
:class:`~xpcsjax.optimization.nlsq.shear_weighting.ShearWeightingConfig`.

Usage
-----

The defence is wired into :func:`xpcsjax.optimization.nlsq.fit_nlsq` and activates
automatically when ``per_angle_mode`` is non-``individual`` and
:math:`N_\phi \geq 3`. A typical invocation is:

.. code-block:: python

   from xpcsjax import fit_nlsq, load_xpcs_data

   data = load_xpcs_data("experiment.hdf5")
   result = fit_nlsq(
       data=data,
       mode="laminar_flow",
       per_angle_mode="auto",   # default
       # all other anti-degeneracy settings inherit dataclass defaults
   )

The fitted parameter vector and the per-angle scaling are stored on
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`. Diagnostics
including the gradient-monitor decisions and the per-angle CV are exposed
through the ``diagnostics`` attribute.

When to use which mode
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Recommended when
   * - ``auto``
     - Default for all production runs (\ :math:`N_\phi \geq 3`).
   * - ``constant``
     - Debugging, or when the quantile estimate is known to be reliable
       and speed matters most.
   * - ``fourier``
     - Genuine smooth azimuthal contrast variation expected (anisotropic
       beam or sample). Tune :math:`K` and verify with information
       criteria.
   * - ``individual``
     - Post-hoc refinement only, initialised from an ``auto`` result.
       Never as a first attempt for :math:`N_\phi > 6`.

.. seealso::

   * :doc:`homodyne_model` -- the laminar-flow kernel that motivates the
     defence.
   * :doc:`heterodyne_model` -- the two-component kernel; Layer 5 is
     gated off in this mode.
   * :doc:`transport_coefficient` -- how :math:`J(t)` enters the residual.
   * :doc:`/advanced/anti_degeneracy` -- engineering-oriented
     companion page covering tuning and diagnostics.
   * :mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller` -- the
     orchestrator.
   * :mod:`xpcsjax.optimization.nlsq.fourier_reparam` -- Layer 1.
   * :mod:`xpcsjax.optimization.nlsq.hierarchical` -- Layer 2.
   * :mod:`xpcsjax.optimization.nlsq.adaptive_regularization` -- Layer 3.
   * :mod:`xpcsjax.optimization.nlsq.gradient_monitor` -- Layer 4.
   * :mod:`xpcsjax.optimization.nlsq.shear_weighting` -- Layer 5.
   * :doc:`citations` -- references for the underlying physics.
