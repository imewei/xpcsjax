.. _theory_heterodyne_model:

Heterodyne Model
================

When a sample contains multiple scattering populations with different
dynamics --- a flowing band and an arrested band, a sample volume and a
reference volume --- the intensity correlation function picks up
cross-correlation contributions whose oscillatory pattern encodes the
relative velocity directly. The ``two_component`` mode in
:class:`xpcsjax.core.HeterodyneModel` implements this generalisation of
the Siegert relation, following the derivation of [He2025]_ (PNAS 2025
SI Section F, Equations S-77 through S-98).

This page derives the two-component kernel, lists the 14 physical parameters
together with the 2 scaling parameters, and describes the multi-angle
dispatch that returns one :class:`~xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult`
per :math:`\phi` angle.

Multi-component scattered field
-------------------------------

For :math:`N` distinguishable scattering components, the total scattered
field at wavevector :math:`\mathbf{q}` and time :math:`t` is (Eq. S-77 of
[He2025]_):

.. math::
   :label: het_field

   E(\mathbf{q}, t) \;=\;
   \sum_{n=1}^{N} x_n(t)\, E_n(\mathbf{q}, t),

where :math:`x_n(t)` is the field amplitude fraction of component
:math:`n` and :math:`E_n(\mathbf{q}, t)` is its scattered field. Each
component has its own transport coefficient :math:`J_n(t)` and mean
velocity :math:`\langle v_n(t)\rangle`.

Two key assumptions (Eq. S-84 of [He2025]_) close the derivation:

1. **Uniform scattering contrast.** All components scatter with the same
   contrast factor, so that intensity fractions are determined solely by the
   composition :math:`x_n(t)`.
2. **No cross-composition spatial correlation.** The positions of
   particles in different components are statistically independent, and
   therefore cross-component field correlations vanish.

General N-component correlation
-------------------------------

Under the two assumptions above, the second-order two-time correlation of
the multi-component intensity is (Eq. S-94 of [He2025]_):

.. math::
   :label: het_cN

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \frac{\beta}{f^2(t_1, t_2)}
   \sum_{n=1}^N \sum_{m=1}^N
       x_n(t_1)\, x_n(t_2)\, x_m(t_1)\, x_m(t_2)\;
       A_{nm}(t_1, t_2),

where the cross-correlation amplitude is

.. math::
   :label: het_Anm

   A_{nm}(t_1, t_2)
   \;=\;
   \exp\!\left(-\tfrac{q^2}{2}\!\int_{t_1}^{t_2}
       \left[J_n(t') + J_m(t')\right] dt'\right)
   \,
   \cos\!\left(q\,\cos\varphi\!\int_{t_1}^{t_2}
       \left[\langle v_n(t')\rangle
            - \langle v_m(t')\rangle\right] dt'\right).

Here :math:`\beta` is the speckle contrast, :math:`\varphi` is the angle
between the velocity and the scattering vector, and the normalisation
factor :math:`f^2` is

.. math::
   :label: het_f2

   f^2(t_1, t_2)
   \;=\;
   \left[\sum_{n=1}^N x_n^2(t_1)\right]
   \left[\sum_{n=1}^N x_n^2(t_2)\right].

**Physical interpretation.** Each :math:`A_{nm}` is the interference between
components :math:`n` and :math:`m`. The cosine factor oscillates when the
two populations drift apart (non-zero relative velocity
:math:`\Delta v_{nm}`), producing characteristic fringes in the :math:`c_2`
matrix. Same-component terms (:math:`n = m`) have zero velocity difference
and their cosine factor is unity, so :math:`A_{nn}` is a pure diffusive
decay.

Two-component (reference + sample) kernel
-----------------------------------------

:class:`xpcsjax.core.HeterodyneModel` implements the :math:`N = 2`
specialisation with a **reference** component (:math:`r`) and a **sample**
component (:math:`s`):

* **Reference** (:math:`r`): static scatterer with transport
  :math:`J_r(t)` and zero mean velocity.
* **Sample** (:math:`s`): moving scatterer with transport :math:`J_s(t)`,
  mean velocity :math:`\langle v(t)\rangle`, and flow angle :math:`\varphi`.

The sample fraction is :math:`x_s(t) \in [0, 1]` and the reference fraction
is :math:`x_r(t) = 1 - x_s(t)`. The two-time correlation (Eq. S-95) is

.. math::
   :label: het_c2_two_comp

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \frac{\beta}{f^2}
   \Bigl[
     \bigl[x_r(t_1)\, x_r(t_2)\bigr]^2 A_{rr}
     \;+\;
     \bigl[x_s(t_1)\, x_s(t_2)\bigr]^2 A_{ss}
     \;+\;
     2\, x_r(t_1)\, x_r(t_2)\, x_s(t_1)\, x_s(t_2)\, A_{rs}
   \Bigr],

with

.. math::

   A_{rr} &= \exp\!\left(-q^2 \!\int_{t_1}^{t_2} J_r(t')\, dt'\right), \\
   A_{ss} &= \exp\!\left(-q^2 \!\int_{t_1}^{t_2} J_s(t')\, dt'\right), \\
   A_{rs} &= \exp\!\left(-\tfrac{q^2}{2}\!\int_{t_1}^{t_2}
                \left[J_r(t') + J_s(t')\right] dt'\right)
            \cos\!\left(q\cos\varphi\!\int_{t_1}^{t_2}
                \langle v(t')\rangle\, dt'\right).

The three contributions admit a clean physical reading:

**Reference self-correlation** -- a monotonic diffusive decay set by the
internal dynamics of the reference scatterers.

**Sample self-correlation** -- a monotonic diffusive decay set by the
sample's transport coefficient; typically faster than the reference because
of flow-enhanced transport.

**Cross-correlation** -- the signature heterodyne term. The cosine factor
produces oscillations whose frequency is proportional to
:math:`q\cos\varphi\,\langle v\rangle`, the projection of the sample
velocity onto the scattering vector. The oscillation amplitude is modulated
by the geometric mean of the transport decays and is maximised when the
reference and sample fractions are balanced (:math:`x_s \approx 0.5`).

Normalisation
-------------

The factor :math:`f^2` in :eq:`het_c2_two_comp` ensures
:math:`c_2(\mathbf{q}, t, t) = 1 + \beta` on the diagonal. For the
two-component system,

.. math::

   f^2(t_1, t_2) \;=\;
     \left[x_s^2(t_1) + x_r^2(t_1)\right]
     \!\cdot\!
     \left[x_s^2(t_2) + x_r^2(t_2)\right].

This accounts for the fact that the total scattered intensity is *not*
simply the sum of individual intensities when the component fractions are
time-dependent.

Angle dependence
----------------

The flow angle :math:`\varphi` controls the projection of velocity onto the
scattering direction:

* :math:`\varphi = 0` (scattering vector parallel to flow): maximum velocity
  sensitivity, :math:`\cos\varphi = 1`.
* :math:`\varphi = \pi/2` (scattering vector perpendicular to flow): zero
  velocity sensitivity, :math:`\cos\varphi = 0`, and the kernel reduces to
  a purely diffusive two-component model.

Measuring at many detector angles therefore samples different projections of
the velocity and constrains the cross-term phase from many independent
observations. xpcsjax exploits this through multi-angle joint fitting; see
:ref:`het_dispatch` below.

Equilibrium projection (one-time form)
--------------------------------------

If the composition fractions, the transport coefficients, and the velocity
are all time-independent, :eq:`het_c2_two_comp` reduces to a function of
the lag :math:`\tau = t_2 - t_1` only. With the equilibrium sample fraction
:math:`x \equiv I_s / (I_s + I_r)` (Eq. S-98 of [He2025]_):

.. math::
   :label: het_g2_eq

   g_2(q, \tau)
   \;=\; 1 + \beta
   \left[
     (1 - x)^2 \exp\!\left(-q^2 \!\int_0^\tau J_r(t')\, dt'\right)
     + x^2 \exp\!\left(-q^2 \!\int_0^\tau J_s(t')\, dt'\right)
     + 2\, x(1 - x)
       \exp\!\left(-\tfrac{q^2}{2}\!\int_0^\tau
         \left[J_r(t') + J_s(t')\right] dt'\right)
       \cos\!\left(q\cos\varphi \!\int_0^\tau
         \langle v(t')\rangle\, dt'\right)
   \right].

.. important::

   xpcsjax **always evaluates the integrals numerically** via cumulative
   trapezoidal integration on the experimental time grid, even when the
   model parameters are time-independent. The package never substitutes the
   closed-form :math:`\int J\, dt = 2 D \tau`. This avoids silent
   approximation errors for the power-law parameterisation
   :math:`J(t) = D_0 t^\alpha + D_\mathrm{offset}`, which has no useful
   antiderivative for generic :math:`\alpha`.

The 14 physics parameters
-------------------------

The ``two_component`` mode separates the dynamics into a reference branch,
a sample branch, the velocity profile, and the composition history. Each
branch has the same power-law-plus-offset structure as the homodyne kernel
(:eq:`hm_Dt` and :eq:`hm_gamma_dot`):

.. list-table:: 14 physical parameters in ``two_component`` mode
   :header-rows: 1
   :widths: 22 22 56

   * - Block
     - Symbol
     - Role
   * - Reference transport
     - :math:`D_{0,r}`
     - Reference diffusion prefactor.
   * -
     - :math:`\alpha_r`
     - Reference diffusion exponent.
   * -
     - :math:`D_{\mathrm{offset},r}`
     - Reference diffusion background.
   * - Sample transport
     - :math:`D_{0,s}`
     - Sample diffusion prefactor.
   * -
     - :math:`\alpha_s`
     - Sample diffusion exponent.
   * -
     - :math:`D_{\mathrm{offset},s}`
     - Sample diffusion background.
   * - Velocity profile
     - :math:`v_0`
     - Velocity prefactor (\ :math:`\mathrm{nm}/\mathrm{s}` or
       :math:`\text{\AA}/\mathrm{s}`, instrument dependent).
   * -
     - :math:`\beta_v`
     - Velocity time exponent.
   * -
     - :math:`v_\mathrm{offset}`
     - Velocity background.
   * -
     - :math:`\varphi`
     - Flow direction in the detector frame (radians).
   * - Composition history
     - :math:`x_{s,0}`
     - Initial sample fraction at :math:`t = 0`.
   * -
     - :math:`r_x`
     - Sample-fraction time-evolution rate.
   * -
     - :math:`x_{s,\infty}`
     - Asymptotic sample fraction.
   * -
     - :math:`q`-encoding factor
     - Velocity-encoding constant absorbing the
       :math:`q \cos\varphi` projection (used by
       :mod:`xpcsjax.core.physics_factors` to precompute
       :math:`q L \Delta t / (2\pi)`).

Together with the 2 per-angle scaling parameters
:math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))` --- handled by the
anti-degeneracy controller exactly as in the homodyne model --- the total
parameter count is **16**.

.. note::

   The velocity-encoding factor on the last row is *fixed* by the
   experimental geometry and the chosen :math:`q`-ring; it is not freely
   optimised. The :mod:`xpcsjax.core.physics_factors` module precomputes
   :math:`q^2 \Delta t / 2` and :math:`q L \Delta t / (2 \pi)`, which appear
   in the diffusion exponent and the velocity-encoding cosine respectively.

.. _het_dispatch:

Multi-angle joint fitting
-------------------------

Heterodyne fits in xpcsjax share the 14 physics parameters across all
:math:`\phi` sectors while allowing the per-angle scaling to vary
according to the chosen anti-degeneracy mode (``auto``, ``constant``,
``fourier``, or ``individual``). The Fourier reparameterisation in
:mod:`xpcsjax.optimization.nlsq.fourier_reparam` expresses
:math:`\beta(\phi)` and :math:`c_\mathrm{offset}(\phi)` as a truncated
Fourier series, which is especially useful for heterodyne because the
oscillatory cross term carries genuine angular signal that the controller
must not smear into per-angle nuisance variation.

The dispatch returns a Python list of
:class:`xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult`, one per :math:`\phi`
angle. Each :class:`~xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult` carries
the optimised parameter vector for that angle, the residual norm, the
trust-region diagnostics, and the per-angle scaling that was used. The
caller is responsible for verifying that the 14 physics parameters agree
across angles (they should, since they are shared in the joint fit).

Oscillatory diagnostics
-----------------------

The presence of oscillatory fringes in the measured :math:`c_2(t_1, t_2)`
at fixed lag :math:`\tau = t_2 - t_1` is a direct signature of
multi-component scattering. The dominant frequency

.. math::

   \nu \;=\; \frac{q\cos\varphi\,|\langle v_n\rangle - \langle v_m\rangle|}{2\pi}

gives the relative velocity between components, from which the individual
velocities can be extracted given the geometry.

* **No oscillations:** single component, fit with
  :class:`xpcsjax.core.HomodyneModel`.
* **One frequency:** two-component (static + flowing) system, fit with
  :class:`xpcsjax.core.HeterodyneModel` (``two_component`` mode).
* **Multiple frequencies:** three or more components or shear banding;
  not implemented in v0.1.

.. warning::

   The heterodyne port is in progress. The public symbol
   :class:`xpcsjax.core.HeterodyneModel` is registered in the lazy export
   table but is currently ``xfail``-marked at the API gate. Treat the
   physics modules under ``xpcsjax.core.heterodyne_*`` as the source of
   truth while the gate flips.

Comparison with the homodyne model
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Feature
     - Homodyne (single component)
     - Heterodyne (two component)
   * - Physical parameters
     - 3 (static) or 7 (laminar flow)
     - 14
   * - Per-angle scaling
     - 2 (constant / averaged) -- 10 (Fourier K=2) -- 2\ :math:`N_\phi`
     - same as homodyne (controller-driven)
   * - :math:`c_2` shape
     - Monotone decay from the diagonal
     - Oscillatory cross fringes superposed on diffusive decays
   * - Applicable to
     - Homogeneous quiescent or laminar-flow sample
     - Static + flowing mixture (reference + sample geometry)
   * - Primary reference
     - [He2024]_
     - [He2025]_

.. seealso::

   * :doc:`homodyne_model` -- single-component kernel.
   * :doc:`correlation_functions` -- Siegert relation and :math:`c_2`.
   * :doc:`transport_coefficient` -- definition of :math:`J(t)`.
   * :doc:`anti_degeneracy` -- per-angle scaling and the five-layer defence.
   * :class:`xpcsjax.core.HeterodyneModel` -- JAX implementation
     (\ ``xfail``-marked at the API gate during the v0.1 port).
   * :doc:`citations` -- including [He2024]_, [He2025]_.
