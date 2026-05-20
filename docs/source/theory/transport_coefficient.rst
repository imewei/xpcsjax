.. _theory_transport_coefficient:

Transport Coefficient J(t)
==================================


The transport coefficient :math:`J(t)` is the central physical quantity in
the homodyne and heterodyne kernels implemented by xpcsjax. It encodes how
the variance of particle displacements grows in time and provides a direct
bridge between microscopic stochastic dynamics and macroscopic rheological
observables through a generalised Green--Kubo relation [He2024]_, [Kubo1966]_.
This page collects the three equivalent definitions of :math:`J(t)`, its
parameterisation in xpcsjax, the convention that links :math:`D_0` to the
Stokes--Einstein coefficient, and the way the resulting integral drives the
NLSQ residual.

Definition
----------

The transport coefficient admits three equivalent representations.

**Variance form** (definition):

.. math::
   :label: tc_J_variance

   J(t) \;=\; \frac{d}{dt}\,\mathrm{Var}\!\left[x(t)\right].

This is the instantaneous rate of growth of the mean-squared displacement
(MSD) of a particle at time :math:`t` ([He2024]_ Eq. S-38). Variance is
taken across an ensemble of particles, or equivalently across the noise
realisation of a Langevin trajectory.

**Covariance form**:

.. math::
   :label: tc_J_covariance

   J(t) \;=\; 2\,\mathrm{Cov}\!\left[x(t),\, v(t)\right],

where :math:`v(t) = \dot{x}(t)` is the instantaneous particle velocity. The
factor of 2 arises from the identity
:math:`\mathrm{Cov}[x, v] = \tfrac{1}{2}\,d\mathrm{Var}[x]/dt`.

**Green--Kubo form** (microscopic origin):

.. math::
   :label: tc_J_greenkubo

   J(t) \;=\; 2\!\int_0^t \mathrm{Cov}\!\left[v(t),\, v(t')\right] dt'.

Equation :eq:`tc_J_greenkubo` is the generalised Green--Kubo relation: the
transport coefficient is twice the integral of the velocity autocorrelation
function (VACF) from :math:`0` to :math:`t`. For a stationary process this
reduces to the classical
:math:`J = 2\!\int_0^\infty C_v(\tau)\, d\tau = 2 D`.

Physical interpretation
-----------------------

The transport coefficient :math:`J(t)` has units of
:math:`\text{length}^2 / \text{time}`, identical to a diffusion coefficient.
It measures how rapidly positional uncertainty accumulates at time
:math:`t`.

.. note::

   In equilibrium, :math:`J` is constant: :math:`J = 2 D`, where :math:`D` is
   the Stokes--Einstein diffusion coefficient. The transport-coefficient
   framework generalises this to **non-stationary** processes where
   :math:`J(t)` varies with time (aging, yielding, transient flow).

The diffusion integral that enters the correlation function (cf.
Equation :eq:`cf_c1_general` of :doc:`correlation_functions`) is

.. math::
   :label: tc_diffusion_integral

   \mathcal{D}(t_1, t_2)
   \;=\; \int_{t_1}^{t_2} J(t')\,dt'
   \;=\; \mathrm{Var}\!\left[x(t_2) - x(t_1)\right].

The right-hand identity follows directly from
:eq:`tc_J_variance`: integrating the variance growth rate over
:math:`[t_1, t_2]` gives the variance of the net displacement. This
identification confirms that :math:`J` directly controls how the
Debye--Waller factor of :math:`c_1` decays.

xpcsjax parameterisation
------------------------

xpcsjax parameterises a time-dependent diffusion rate as a power law with
offset,

.. math::
   :label: tc_D_model

   D(t) \;=\; D_0 \cdot t^\alpha + D_\mathrm{offset},

where

* :math:`D_0 > 0` is the diffusion prefactor;
* :math:`\alpha \in (-1, 1]` is the anomalous exponent
  (:math:`\alpha = 0` Brownian, :math:`\alpha < 0` sub-diffusive aging,
  :math:`\alpha > 0` super-diffusive);
* :math:`D_\mathrm{offset} \geq 0` is a constant background diffusion.

The diffusion integral entering :math:`c_1` (and therefore :math:`c_2`
through the Siegert relation) is computed by cumulative trapezoidal
integration on the experimental time grid:

.. math::
   :label: tc_integral_xpcsjax

   \mathcal{D}(t_1, t_2)
   \;=\; \int_{t_1}^{t_2} D(t')\, dt'.

The trapezoidal kernel and the precomputed geometric factors
:math:`q^2 \Delta t / 2` and :math:`q L \Delta t / (2\pi)` are exposed by
:mod:`xpcsjax.core.physics_factors`. The same numerical kernel is reused by
both :class:`xpcsjax.core.HomodyneModel` and
:class:`xpcsjax.core.HeterodyneModel`.

.. important::

   The closed-form antiderivative
   :math:`\int_0^\tau D_0 t'^\alpha\, dt' = D_0 \tau^{\alpha+1}/(\alpha+1)`
   exists only for :math:`\alpha \neq -1` and produces noticeable error
   relative to trapezoidal integration on the actual time grid when
   :math:`D_\mathrm{offset}` is non-zero or the grid is irregular. xpcsjax
   never substitutes the closed form, even when the parameters happen to be
   time-independent.

The D_0 = 2 D_SE convention
--------------------------------------------


The xpcsjax (and homodyne / heterodyne) parameter :math:`D_0` absorbs a
factor of 2 from the formal transport coefficient. For standard Brownian
motion, the physical Stokes--Einstein diffusion coefficient is

.. math::

   D_\mathrm{SE} \;=\; \frac{k_B T}{6 \pi \eta R_h},

while xpcsjax stores :math:`D_0 = 2 D_\mathrm{SE}`. The reason is the
Siegert relation :eq:`cf_siegert`: the measured :math:`c_2` depends on
:math:`|c_1|^2 = \exp(-q^2 \mathcal{D})` rather than on :math:`c_1` itself.
For the textbook equilibrium result
:math:`|c_1|^2 = \exp(-2 q^2 D_\mathrm{SE}\,\tau)` to hold, the integral
:math:`\mathcal{D} = \int D(t')\,dt'` must equal :math:`2 D_\mathrm{SE}\,\tau`,
which requires :math:`D_0 = 2 D_\mathrm{SE}`.

.. warning::

   When comparing :math:`D_0` from xpcsjax to a Stokes--Einstein estimate
   computed from particle radius and solvent viscosity, divide by 2.
   Equivalently, when reporting :math:`D_\mathrm{SE}` to a rheology
   audience, use :math:`D_\mathrm{SE} = D_0 / 2`.

Mean-squared displacement
-------------------------

In the xpcsjax parameterisation (where :math:`D_0` absorbs the factor of 2;
see warning above), the mean-squared displacement is

.. math::
   :label: tc_msd

   \mathrm{MSD}(t)
   \;\equiv\; \mathrm{Var}\!\left[x(t) - x(0)\right]
   \;=\; \int_0^t D(t')\, dt'.

The effective physical (Stokes--Einstein) diffusion coefficient at time
:math:`t` is therefore half the xpcsjax integrand averaged over time:

.. math::

   D_\mathrm{SE}(t) \;=\; \frac{\mathrm{MSD}(t)}{2 t}.

For :math:`\alpha = 0` (standard Brownian motion), :math:`D_\mathrm{SE}` is
constant. For :math:`\alpha < 0` (sub-diffusion with aging), particles slow
down over time.

Transport coefficient for classical processes
---------------------------------------------

The following table summarises :math:`J(t)` for Langevin processes that
arise naturally in soft matter; in each row :math:`D` is the physical
Stokes--Einstein diffusion coefficient (xpcsjax uses :math:`D_0 = 2 D`).

.. list-table:: :math:`J(t)` for canonical Langevin processes
   :header-rows: 1
   :widths: 30 40 30

   * - Process
     - :math:`J(t)`
     - Regime
   * - Wiener (free diffusion)
     - :math:`2 D`
     - :math:`D` = constant
   * - Anomalous diffusion
     - :math:`2 D_0\, t^{\alpha}`
     - :math:`\alpha \in (-1, 1]`
   * - Ornstein--Uhlenbeck
     - :math:`2 D\!\left(1 - e^{-\gamma t}\right)^{2}`
     - Confinement radius
       :math:`\sqrt{D/\gamma}` [UhlenbeckOrnstein1930]_
   * - Brownian oscillator (underdamped)
     - :math:`2 D (\gamma^2 / \omega_s^2) e^{-\gamma t} \sin^2(\omega_s t)`
     - :math:`\omega_s^2 = \omega_0^2 - \gamma^2/4 > 0`
   * - Brownian oscillator (overdamped)
     - :math:`8 D (\gamma^2 / \gamma_s^2) e^{-\gamma t} \sinh^2(\gamma_s t / 2)`
     - :math:`\gamma_s^2 = \gamma^2 - 4\omega_0^2 > 0`

Here :math:`\gamma` is the friction coefficient and :math:`\omega_0` is the
trap frequency, with :math:`\omega_s = \sqrt{\omega_0^2 - \gamma^2/4}` and
:math:`\gamma_s = \sqrt{\gamma^2 - 4\omega_0^2}`.

Why J(t) drives the residual
------------------------------------


In each homodyne mode and in the heterodyne two-component kernel,
:math:`J(t)` enters the forward model *only* through the cumulative
integral :eq:`tc_diffusion_integral`. The NLSQ residual at each
:math:`(\phi_k, t_i, t_j)` therefore has the form

.. math::

   r_{kij}(\theta)
   \;=\;
   c_2^{kij,\,\mathrm{meas}}
   - c_\mathrm{offset}(\phi_k)
   - \beta(\phi_k)\, e^{-q^2 \mathcal{D}(t_i, t_j;\, \theta)}
     \, M(\phi_k, t_i, t_j;\, \theta),

where :math:`M` is the mode-specific shear modulation (sinc\ :sup:`2` in
laminar flow, identity in static modes, oscillatory cosine cross term in
``two_component``). The Jacobian of the residual with respect to
:math:`(D_0, \alpha, D_\mathrm{offset})` propagates entirely through the
cumulative trapezoidal integral; the upstream NLSQ library handles this
through forward-mode automatic differentiation of the JAX-traced forward
model.

Three consequences follow:

1. The exponential of the integral means that absolute errors on
   :math:`D_0` translate to relative errors on :math:`c_2` that scale with
   :math:`q^2 \mathcal{D}`. Far from the diagonal,
   :math:`q^2 \mathcal{D}` is large, the residual saturates, and the fit
   becomes insensitive to further increases of :math:`D_0`. The informative
   data live close to the diagonal.
2. With :math:`\alpha < 0`, the integrand diverges at :math:`t = 0`. The
   trapezoidal rule integrates from the first available frame, and xpcsjax
   relies on the data loader to start the time grid at a strictly positive
   value (typically the integration time of the first frame).
3. Because the integral is monotone non-decreasing in :math:`D_0` and is
   evaluated by a vectorised cumulative-trapezoid kernel, the Jacobian is
   dense and well-behaved. This is precisely the property that the
   Levenberg--Marquardt step in NLSQ exploits, and it is why xpcsjax
   delegates the trust-region solve to the upstream library while keeping
   strategy in its own anti-degeneracy controller (see
   :doc:`anti_degeneracy`).

Connection to rheology
----------------------

The Green--Kubo form :eq:`tc_J_greenkubo` links :math:`J(t)` to the
complex shear modulus :math:`G^{*}(\omega)` of the suspending medium
through the generalised Stokes--Einstein relation,

.. math::

   D(\omega)
   \;=\;
   \frac{k_B T}{6 \pi R \,\eta(\omega)}
   \;=\;
   \frac{k_B T}{6 \pi R}\,\frac{1}{G^{*}(\omega)},

where :math:`R` is the particle radius and :math:`\eta(\omega)` is the
frequency-dependent viscosity. Measuring :math:`J(t)` from XPCS data
therefore provides a non-invasive probe of local viscoelastic properties.

For the yielding transition studied in [He2025]_, the time evolution of
:math:`J(t)` during the rheological loading protocol distinguishes:

* **Repulsive suspensions:** Andrade creep
  (\ :math:`\gamma \sim t^{1/3}`) maps to
  :math:`J(t) \propto t^{-2/3}` --- a sub-diffusive, aging transport
  coefficient [Andrade1910]_.
* **Attractive suspensions:** heterogeneous shear banding produces
  non-Gaussian displacement distributions not captured by a single
  :math:`J(t)`, motivating the heterodyne two-component model.

.. seealso::

   * :doc:`correlation_functions` -- how :math:`J(t)` enters :math:`c_1`
     and :math:`c_2`.
   * :doc:`homodyne_model` -- the kernels that use :eq:`tc_D_model`.
   * :doc:`heterodyne_model` -- the two-branch generalisation
     :math:`J_r(t), J_s(t)`.
   * :doc:`anti_degeneracy` -- the per-angle defence that keeps
     :math:`D_0` identifiable when the shear gradient cancels across
     angles.
   * :mod:`xpcsjax.core.physics_factors` -- precomputed geometric
     prefactors and trapezoidal kernel.
   * :doc:`citations` -- references including [He2024]_, [He2025]_,
     [Kubo1966]_, [UhlenbeckOrnstein1930]_, [Andrade1910]_.
