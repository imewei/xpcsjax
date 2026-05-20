.. _theory_homodyne_model:

Homodyne Model
==============

The homodyne model in xpcsjax describes a single scattering population whose
intensity correlation function follows the Siegert relation
:eq:`cf_siegert`. The class :class:`xpcsjax.core.HomodyneModel` provides
four analysis modes selected by the ``mode`` argument:

* ``static`` -- single-component quiescent sample,
* ``static_isotropic`` -- static sample treated as :math:`\phi`-averaged,
* ``static_anisotropic`` -- static sample with angle-resolved dynamics,
* ``laminar_flow`` -- shear-flow geometry with gap integration.

This page derives the per-mode kernels, lists the parameter sets, and shows
how :meth:`xpcsjax.core.HomodyneModel.compute_c2` assembles the two-time
prediction. The development follows [He2024]_; the heterodyne (mixed
population) generalisation is covered separately in
:doc:`heterodyne_model`.

Common ingredients
------------------

All homodyne modes share three building blocks.

**Diffusion integral.** With the power-law parameterisation

.. math::
   :label: hm_Dt

   D(t) \;=\; D_0 \cdot t^\alpha + D_\mathrm{offset},

the diffusion integral entering the Siegert relation is

.. math::
   :label: hm_diff_integral

   \mathcal{D}(t_1, t_2) \;=\; \int_{t_1}^{t_2} D(t')\,dt',

evaluated by cumulative trapezoidal integration on the experimental time
grid. The :math:`D_0 = 2 D_\mathrm{SE}` convention is detailed in
:doc:`transport_coefficient`.

**Siegert kernel.** From :eq:`cf_siegert`,

.. math::
   :label: hm_kernel

   c_2(\phi, t_1, t_2)
   \;=\; c_\mathrm{offset}(\phi)
   \;+\; \beta(\phi)\, |g_1(\phi, t_1, t_2)|^2,

with :math:`|g_1|^2` mode-dependent.

**Precomputed factors.** :mod:`xpcsjax.core.physics_factors` precomputes the
geometric prefactors :math:`q^2 \Delta t / 2` and
:math:`q\, L\, \Delta t / (2\pi)` used by the integrand evaluation; this
amortises the per-call cost across the inner JAX loop.

Static modes
------------

In the static modes there is no flow and the mean velocity vanishes, so the
external phase in :eq:`cf_c1_general` is identically one. The Siegert
relation reduces to

.. math::
   :label: hm_c2_static

   c_2(\phi, t_1, t_2)
   \;=\; c_\mathrm{offset}(\phi)
   \;+\; \beta(\phi)\,
         \exp\!\left(-q^2\,\mathcal{D}(t_1, t_2)\right).

The three static sub-modes differ only in how they treat the
:math:`\phi`-dependence of the scaling parameters.

``static`` mode
~~~~~~~~~~~~~~~

Treats the full set of azimuthal sectors as independent samples of one
isotropic kernel. The kernel itself does not depend on :math:`\phi`; only the
nuisance scaling parameters :math:`(\beta(\phi), c_\mathrm{offset}(\phi))`
do. The physical parameter vector has three entries:

.. math::

   \theta_\mathrm{static} = (D_0, \alpha, D_\mathrm{offset}).

``static_isotropic`` mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Assumes the per-angle nuisance parameters are uniform across :math:`\phi`
(single :math:`\beta` and single offset). This is the smallest model in the
homodyne family and is the recommended starting point for quiescent samples
when no per-angle inhomogeneity is expected. The physical parameter vector
is again

.. math::

   \theta_\mathrm{static\_iso} = (D_0, \alpha, D_\mathrm{offset}),

and the per-angle scaling collapses to a single
:math:`(\bar{\beta}, \bar{c}_\mathrm{offset})` pair.

``static_anisotropic`` mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keeps :math:`\beta(\phi)` and :math:`c_\mathrm{offset}(\phi)` independent
per sector (handled through the per-angle modes in
:doc:`anti_degeneracy`), but the underlying physics is still purely
diffusive. Use this mode when the sample is static but the contrast varies
with angle for instrumental reasons.

Laminar flow mode
-----------------

In a Taylor--Couette or shear-cell geometry, the suspending fluid imposes a
systematic drift on every particle. The X-ray beam traverses the sample at
an azimuthal angle :math:`\phi` relative to the flow direction and samples
the velocity distribution across the gap, generating a sinc\ :sup:`2`
modulation in :math:`c_2`.

Gap integration and the sinc-squared modulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a position :math:`x` inside the gap (:math:`0 \leq x \leq h`) the local
flow velocity in simple shear is :math:`v_x(x, t) = \dot{\gamma}(t)\,x`.
Projecting onto the scattering vector at azimuthal angle :math:`\phi`,

.. math::

   v_\parallel(x, t) \;=\; \dot{\gamma}(t)\,x\,\cos\phi.

The shear contribution to :math:`c_1` is a phase factor
:math:`e^{i q \!\int v_\parallel \, dt}`. Integrating uniformly over the gap,

.. math::
   :label: hm_gap_int

   c_1^{(\mathrm{shear})}(\mathbf{q}, t_1, t_2)
   \;=\;
   \frac{1}{h}\int_0^h
     \exp\!\left(i\, q\cos\phi \int_{t_1}^{t_2}\dot{\gamma}(t)\,x\,dt\right) dx,

and writing :math:`\Gamma(t_1, t_2) = \int_{t_1}^{t_2}\dot{\gamma}(t)\,dt`
for the accumulated strain,

.. math::

   \frac{1}{h}\int_0^h e^{i\, q\cos\phi\,\Gamma\, x}\,dx
   \;=\;
   e^{i\,\tfrac{q h \cos\phi\,\Gamma}{2}}
   \cdot
   \mathrm{sinc}\!\left(\tfrac{q h \cos\phi\,\Gamma}{2\pi}\right).

The phase factor cancels under :math:`|\cdot|^2`, leaving

.. math::
   :label: hm_sinc2

   \left|c_1^{(\mathrm{shear})}\right|^2
   \;=\;
   \mathrm{sinc}^2\!\left(\tfrac{q h \cos\phi\,\Gamma(t_1, t_2)}{2\pi}\right).

.. note::

   xpcsjax follows the NumPy convention
   :math:`\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)`, so :math:`\mathrm{sinc}(0) = 1`.

Full laminar-flow kernel
~~~~~~~~~~~~~~~~~~~~~~~~

Combining the diffusive Debye--Waller decay :eq:`cf_c2_homodyne` with the
shear modulation :eq:`hm_sinc2`,

.. math::
   :label: hm_c2_laminar

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; c_\mathrm{offset}(\phi)
   \;+\; \beta(\phi)\,
       \exp\!\left(-q^2\!\int_{t_1}^{t_2} J(t')\,dt'\right)
       \,
       \mathrm{sinc}^2\!\left(\tfrac{q h \cos(\phi - \phi_0)\,\Gamma(t_1, t_2)}{2\pi}\right),

where:

* :math:`\Gamma(t_1, t_2) = \int_{t_1}^{t_2}\dot{\gamma}(t)\,dt` is the
  accumulated strain;
* :math:`\phi_0` maps the laboratory angle to the physical angle between
  scattering vector and flow direction;
* :math:`h` is the rheometer gap, fixed by the instrument geometry and
  *not* a fitted parameter.

.. warning::

   Equation :eq:`hm_c2_laminar` is valid only for homodyne detection (single
   scattering population). If the sample contains multiple populations with
   different mean velocities, use the heterodyne kernel in
   :doc:`heterodyne_model`.

Shear-rate parameterisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

xpcsjax parameterises the shear rate as a power law with offset,

.. math::
   :label: hm_gamma_dot

   \dot{\gamma}(t)
   \;=\;
   \dot{\gamma}_0 \cdot t^{\beta_\gamma} + \dot{\gamma}_\mathrm{offset},

where :math:`\dot{\gamma}_0` is the prefactor, :math:`\beta_\gamma` is the
time exponent (distinct from the speckle contrast :math:`\beta` --- context
disambiguates), and :math:`\dot{\gamma}_\mathrm{offset}` is a constant
background. The accumulated strain :math:`\Gamma(t_1, t_2)` is computed by
cumulative trapezoidal integration on the experimental time grid, matching
the diffusion integral convention.

Angular dependence
~~~~~~~~~~~~~~~~~~

The :math:`\cos(\phi - \phi_0)` factor makes the laminar-flow kernel
strongly angle-dependent:

* :math:`\phi = \phi_0 \pm \pi/2` (scattering vector perpendicular to flow):
  :math:`\cos = 0`, the sinc\ :sup:`2` term equals one and only the diffusive
  decay contributes.
* :math:`\phi = \phi_0` or :math:`\phi = \phi_0 + \pi` (parallel/anti-parallel
  to flow): :math:`|\cos| = 1`, the sinc\ :sup:`2` modulation is maximal.
* Intermediate angles: mixed contribution.

This anisotropy is the lever that allows xpcsjax to extract
:math:`\dot{\gamma}_0` --- and the principal reason the anti-degeneracy
controller exists. See :doc:`anti_degeneracy` for the angle-aware scaling
and shear-sensitivity weighting.

Parameter sets per mode
-----------------------

.. list-table:: Physical parameter vectors in each homodyne mode
   :header-rows: 1
   :widths: 28 18 54

   * - Mode
     - Symbol
     - Role
   * - ``static`` (3 params)
     - :math:`D_0`
     - Diffusion prefactor (\ :math:`\text{\AA}^2 / \mathrm{s}` units).
   * -
     - :math:`\alpha`
     - Anomalous exponent (:math:`\alpha = 0` Brownian, :math:`< 0` sub-diffusive).
   * -
     - :math:`D_\mathrm{offset}`
     - Constant diffusion background.
   * - ``static_isotropic`` (3 params)
     - same
     - As ``static``; per-angle scaling forced uniform.
   * - ``static_anisotropic`` (3 params)
     - same
     - As ``static``; per-angle scaling treated per
       :doc:`anti_degeneracy` mode.
   * - ``laminar_flow`` (7 params)
     - :math:`D_0`
     - Diffusion prefactor.
   * -
     - :math:`\alpha`
     - Diffusion exponent.
   * -
     - :math:`D_\mathrm{offset}`
     - Constant diffusion background.
   * -
     - :math:`\dot{\gamma}_0`
     - Shear-rate prefactor (\ :math:`\mathrm{s}^{-1}`).
   * -
     - :math:`\beta_\gamma`
     - Shear-rate time exponent.
   * -
     - :math:`\dot{\gamma}_\mathrm{offset}`
     - Constant shear-rate background.
   * -
     - :math:`\phi_0`
     - Flow-direction offset relative to the detector frame (radians).

Forward API
-----------

The class :class:`xpcsjax.core.HomodyneModel` exposes the kernel through
:meth:`xpcsjax.core.HomodyneModel.compute_c2`:

.. code-block:: python

   from xpcsjax import HomodyneModel
   import jax.numpy as jnp

   model = HomodyneModel(mode="laminar_flow", q=q, h=h_gap)

   params = jnp.array([
       D0, alpha, D_offset,
       gamma_dot_0, beta_gamma, gamma_dot_offset, phi_0,
   ])

   c2 = model.compute_c2(
       params=params,
       phi_angles=phi_angles,   # shape (n_phi,)
       contrast=0.5,            # scalar β
       offset=1.0,              # scalar baseline
   )                            # → (n_phi, n_time, n_time)

The signature is

.. code-block:: python

   HomodyneModel.compute_c2(
       params, phi_angles, contrast=0.5, offset=1.0
   ) -> jnp.ndarray  # shape (n_phi, n_time, n_time)

For ``static``/``static_isotropic``/``static_anisotropic`` the ``params``
vector is truncated to the 3-entry diffusion block. The ``contrast`` and
``offset`` arguments are scalars when the anti-degeneracy controller uses
``constant`` or ``auto``-averaged scaling, and per-angle arrays when it uses
``individual`` or ``fourier``. The pipeline that drives this dispatch lives
in :mod:`xpcsjax.optimization.nlsq`.

Connection to rheology
----------------------

The Andrade creep law :math:`\gamma \sim t^{1/3}` observed in repulsive
colloidal suspensions near the glass transition [He2025]_ corresponds to

.. math::

   \dot{\gamma}(t) \;\propto\; t^{-2/3}
   \quad\Longrightarrow\quad
   \beta_\gamma = -\tfrac{2}{3},
   \qquad \dot{\gamma}_0 > 0,

which is the canonical signature of plastic creep driven by stress-activated
cage rearrangements. By fitting :eq:`hm_c2_laminar` to two-time XPCS data,
xpcsjax produces a direct microscopic probe of bulk rheological behaviour.

Why the laminar-flow fit is hard
--------------------------------

The combination of (i) a power-law diffusion factor that contributes
isotropically across :math:`\phi` and (ii) a sinc\ :sup:`2` shear factor
whose gradient depends on :math:`\cos(\phi - \phi_0)` and changes sign over
:math:`[0, 2\pi)` creates a parameter absorption degeneracy when per-angle
scaling parameters are free. Treating each
:math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))` as an independent free
parameter gives the optimiser :math:`2 N_\phi + 7` degrees of freedom, and
the shear gradient cancels when summed over angles.

The five-layer defence in :mod:`xpcsjax.optimization.nlsq.anti_degeneracy_controller`
exists for exactly this reason; see :doc:`anti_degeneracy`.

.. seealso::

   * :doc:`correlation_functions` -- :math:`g_1`, :math:`g_2`, Siegert relation.
   * :doc:`transport_coefficient` -- definition of :math:`J(t)` and the
     :math:`D_0 = 2 D_\mathrm{SE}` convention.
   * :doc:`heterodyne_model` -- two-component (mixed-population) kernel.
   * :doc:`anti_degeneracy` -- five-layer defence on top of the kernel.
   * :class:`xpcsjax.core.HomodyneModel` -- JAX implementation.
   * :func:`xpcsjax.optimization.nlsq.fit_nlsq` -- the NLSQ fitting entry point.
   * :doc:`citations` -- including [He2024]_, [He2025]_, [Sutton2008]_.
