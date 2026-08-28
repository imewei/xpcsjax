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

Layer 1 -- Per-angle reparameterisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module:** :mod:`xpcsjax.optimization.nlsq.per_angle_mode`.
**Class:** :class:`~xpcsjax.optimization.nlsq.per_angle_mode.PerAngleScalingPlan`
(resolved by :func:`~xpcsjax.optimization.nlsq.per_angle_mode.resolve_per_angle_mode`).

This layer attacks the structural degeneracy by reducing the dimension of
the per-angle scaling space. Three resolved modes are available, selected
through the ``per_angle_mode`` setting (the ``auto`` token resolves to one of
them):

* ``constant`` -- per-angle :math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))`
  are estimated from data quantiles and held **fixed** during the fit. Only
  the physical parameters are optimised. Total parameters: physical only.
* ``averaged`` -- computes the quantile estimates, averages them to a single
  :math:`(\bar{\beta}, \bar{c}_\mathrm{offset})`, and optimises these two
  averaged scalars together with the physical parameters. This is what
  ``auto`` (the default) resolves to for :math:`N_\phi \geq 3`.
* ``individual`` -- each angle has independent
  :math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))`, adding
  :math:`2 N_\phi` free parameters. This is what ``auto`` resolves to for
  :math:`N_\phi < 3`; also usable as a post-hoc refinement of an
  ``averaged``-mode fit.

The quantile estimation underlying ``constant`` and ``averaged`` exploits the
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
   * - ``averaged``
     - 9
     - 7 physical + 2 averaged scaling; what ``auto`` (the recommended
       default) resolves to for :math:`N_\phi \geq 3`.
   * - ``individual``
     - 53
     - 7 physical + 46 per-angle; high degeneracy risk. ``auto`` resolves
       to this for :math:`N_\phi < 3`.

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
declared and recorded.

.. important::

   **Layer 4 is strictly observational in the wired solve path.** The
   per-iteration callback actually passed to the NLSQ solver
   (:func:`~xpcsjax.optimization.nlsq.gradient_monitor.build_gradient_collapse_callback`)
   feeds the monitor and always returns ``None`` -- monitor-on and
   monitor-off produce a bit-identical fit trajectory. ``response_mode``
   (``"warn"`` / ``"hierarchical"`` / ``"reset"`` / ``"abort"``, default
   ``"hierarchical"``) configures what :meth:`GradientCollapseMonitor.get_response`
   *would* recommend, and it is surfaced in diagnostics for a human or a
   post-hoc caller to act on -- but no production call site currently
   invokes ``get_response()`` to change the running solve. Treat collapse
   detection as a diagnostic signal (log it, inspect
   ``nlsq_diagnostics["gradient_monitor"]``), not as an active intervention.

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

   Layer 5 is **gated to** ``laminar_flow`` **only**. The
   :math:`|\cos(\phi - \phi_0)|` weighting is derived from the laminar shear
   gradient :math:`\partial g_1 / \partial \dot{\gamma}_0 \propto
   \cos(\phi - \phi_0)`, so it is meaningful only where a shear rate appears
   in the kernel:

   * ``static_isotropic`` / ``static_anisotropic`` --- no flow direction and
     no shear-sensitivity peak, so L5 is gated off (the static degeneracy is
     handled structurally by Layers 1--4).
   * ``two_component`` (heterodyne) --- has its **own** velocity/flow term
     (:math:`v_0`, :math:`v_\mathrm{offset}`, :math:`\phi_{0,\mathrm{het}}`),
     but it is *structurally different* from laminar_flow's shear rate, so the
     laminar weighting does not transfer; the angular information is already
     well distributed across :math:`\phi`.

   The gating is declared in ``_LAYER_GATES`` inside
   :mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`. Note that
   ``is_layer_active()`` returns ``True`` for every layer when
   ``analysis_mode`` is ``None`` (the homodyne characterisation gate's path),
   so the gating does not affect the ``rtol=1e-10`` parity baselines.

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

Layer activation by analysis mode
---------------------------------

Only Layer 5 is gated by ``analysis_mode``; Layers 1--4 are available in every
mode (their *effect* still depends on configuration and per-angle mode). The
following table shows which layers a mode can run (the hard gate), independent
of the config-level enable flags:

.. list-table:: Layer availability by analysis mode (the gate)
   :header-rows: 1
   :widths: 28 12 12 12 12 12

   * - Mode
     - L1
     - L2
     - L3
     - L4
     - L5
   * - ``static_isotropic``
     - yes
     - yes
     - yes
     - yes
     - **no**
   * - ``static_anisotropic``
     - yes
     - yes
     - yes
     - yes
     - **no**
   * - ``laminar_flow``
     - yes
     - yes
     - yes
     - yes
     - **yes**
   * - ``two_component``
     - yes
     - yes
     - yes
     - yes
     - **no**

Recommended per-mode setup (template defaults)
----------------------------------------------

The four shipped templates under ``xpcsjax/config/templates/`` carry the
maintainer-tuned defaults below. They optimise for **fit robustness**; the
``two_component`` defaults in particular run an extra full solve (L2) and can
be relaxed when wall-time matters and the fit is well behaved.

.. list-table:: Recommended anti-degeneracy configuration per mode
   :header-rows: 1
   :widths: 18 16 13 13 18 13

   * - Mode
     - ``per_angle_mode``
     - L2
     - L3
     - L4 response
     - L5
   * - ``static_isotropic``
     - ``constant`` (fixed)
     - off
     - off
     - ``warn``
     - n/a
   * - ``static_anisotropic``
     - ``auto``
     - off
     - off
     - ``warn``
     - n/a
   * - ``laminar_flow``
     - ``auto``
     - **on**
     - off
     - ``hierarchical``
     - **on**
   * - ``two_component``
     - ``auto``
     - **on**
     - off
     - ``hierarchical``
     - n/a

Notes on the recommendations:

* **L1** is always the primary defense. For ``static_isotropic`` it is the
  *only* one that matters (``per_angle_mode: "constant"`` freezes scaling; the
  3-parameter problem is unimodal).
* **L2 and L4 are configured together.** ``gradient_response_mode:
  "hierarchical"`` records that collapse *should* escalate into Layer 2 --
  but see the Layer 4 note above: the production callback never actually
  triggers this escalation, so pairing them is a documentation convention
  (and a hook for a human/future caller reading the diagnostics) rather
  than a live runtime interaction. ``laminar_flow`` and ``two_component``
  pair ``hierarchical`` with L2 on; the static modes pair ``warn`` with L2
  off.
* **L2 is mandatory for L5.** In ``laminar_flow`` the alternating
  frozen-scaling / frozen-physics solve breaks the absorption coupling so the
  shear weighting can take effect; the template flags ``hierarchical.enable``
  as *critical*.
* **L3 is inert in** ``auto``-**averaged mode.** With a single shared
  ``(contrast, offset)`` pair there is only one scaling group, so the
  cross-group CV penalty is identically zero. L3 only bites once there are
  :math:`\geq 2` groups (``individual``).
* **CMA-ES** (a separate escape path, not a layer) is rarely needed for static
  modes, often needed for ``laminar_flow``, and routinely engaged for
  ``two_component`` (14-D, wide parameter scales).

Configuration
-------------

The controller is configured by an
:class:`~xpcsjax.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig`
dataclass, built via ``AntiDegeneracyConfig.from_dict()`` from the mode
YAML's ``anti_degeneracy:`` block. **The YAML shape is nested** (a
sub-mapping per layer); the flat ``layer_field`` names below are the
*Python dataclass attribute*, not the YAML key:

.. code-block:: yaml

   anti_degeneracy:
     enable: true                         # -> AntiDegeneracyConfig.enable
     per_angle_mode: "auto"               # "individual" | "constant" | "averaged" | "auto"
     constant_scaling_threshold: 3        # Nphi cutover: auto -> "averaged" at n_phi >= threshold, else "individual"
     execute_layers: false                # opt-in L2/L3 escape gate on the >=1M stratified-LS path only

     hierarchical:                        # -> hierarchical_* fields
       enable: true                       # hierarchical_enable
       max_outer_iterations: 5            # hierarchical_max_outer_iterations
       outer_tolerance: 1.0e-6            # hierarchical_outer_tolerance
       physical_max_iterations: 100       # hierarchical_physical_max_iterations
       per_angle_max_iterations: 50       # hierarchical_per_angle_max_iterations

     regularization:                      # -> regularization_* fields
       enable: false
       mode: "relative"                   # regularization_mode: "absolute" | "relative" | "auto"
       lambda: 1.0                        # regularization_lambda
       target_cv: 0.10                    # regularization_target_cv
       target_contribution: 0.10          # regularization_target_contribution
       max_cv: 0.20                       # regularization_max_cv
       auto_tune_lambda: true             # regularization_auto_tune_lambda

     gradient_monitoring:                 # -> gradient_* fields
       enable: true                       # gradient_monitoring_enable
       ratio_threshold: 0.01              # gradient_ratio_threshold
       consecutive_triggers: 5            # gradient_consecutive_triggers
       response: "hierarchical"           # gradient_response_mode -- see the Layer 4 note above: not currently wired to change the running solve

     shear_weighting:                     # -> shear_weighting_* fields (laminar_flow only)
       enable: true
       min_weight: 0.3
       alpha: 1.0

``enable``, ``per_angle_mode``, ``constant_scaling_threshold``, and
``execute_layers`` are the only top-level (non-nested) keys; every other
layer's settings live under its own sub-mapping. The full set of fields
and their nested-key mapping is enumerated in
:meth:`AntiDegeneracyConfig.from_dict()
<xpcsjax.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.from_dict>`.

Usage
-----

The defence is wired into :func:`xpcsjax.optimization.nlsq.fit_nlsq` and activates
automatically when ``per_angle_mode`` is non-``individual`` and
:math:`N_\phi \geq 3`. A typical invocation is:

.. code-block:: python

   from xpcsjax import fit_nlsq, load_xpcs_data

   data = load_xpcs_data("experiment.hdf5")
   # analysis_mode ("laminar_flow") and per_angle_mode ("auto", the default)
   # are set in the config (ConfigManager / YAML), not passed as kwargs.
   result = fit_nlsq(data, config)

The fitted parameter vector and the per-angle scaling are stored on
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`. Diagnostics
including the gradient-monitor decisions and the per-angle CV are exposed
through the ``nlsq_diagnostics`` attribute (streaming / out-of-core and
stratified paths additionally populate ``streaming_diagnostics`` /
``stratification_diagnostics``).

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
   * - ``averaged``
     - A single shared contrast/offset is adequate and you want it
       optimised rather than frozen; what ``auto`` resolves to for
       :math:`N_\phi \geq 3`.
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
   * :mod:`xpcsjax.optimization.nlsq.per_angle_mode` -- Layer 1.
   * :mod:`xpcsjax.optimization.nlsq.hierarchical` -- Layer 2.
   * :mod:`xpcsjax.optimization.nlsq.adaptive_regularization` -- Layer 3.
   * :mod:`xpcsjax.optimization.nlsq.gradient_monitor` -- Layer 4.
   * :mod:`xpcsjax.optimization.nlsq.shear_weighting` -- Layer 5.
   * :doc:`citations` -- references for the underlying physics.
