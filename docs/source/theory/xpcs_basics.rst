.. _theory_xpcs_basics:

XPCS Basics
===========

X-ray Photon Correlation Spectroscopy (XPCS) is a coherent scattering technique
that resolves the dynamics of soft and complex materials by tracking the
temporal evolution of the far-field speckle pattern produced by a partially
coherent X-ray beam. This page introduces the quantities measured by an XPCS
experiment, how they connect to the underlying particle dynamics, and how the
raw data flow into the xpcsjax fitting pipeline.

Scattered field, intensity, and speckle
---------------------------------------

For a sample of :math:`N` scatterers at positions :math:`\mathbf{r}_j(t)`, the
position density in Fourier space is

.. math::
   :label: xb_rho

   \rho(\mathbf{q}, t) \;=\; \sum_{j=1}^N f_j
       \exp\!\left(i\,\mathbf{q}\cdot\mathbf{r}_j(t)\right),

where :math:`f_j` is the form factor of particle :math:`j` and the momentum
transfer is :math:`|\mathbf{q}| = q = 4\pi \sin(\theta)/\lambda`. The
scattered electric field is :math:`E(\mathbf{q}, t) \propto \rho(\mathbf{q}, t)`
and the detected intensity is

.. math::
   :label: xb_intensity

   I(\mathbf{q}, t) \;=\; |E(\mathbf{q}, t)|^2.

The interference of partial waves from a coherent illumination produces a
high-contrast speckle pattern whose grains fluctuate as the scatterers
rearrange. Each detector pixel samples one realisation of these fluctuations,
giving a time series :math:`I(\mathbf{q}, t)` per pixel.

.. note::

   A practical detector contains :math:`10^5`--:math:`10^7` pixels grouped by
   :math:`q`-magnitude and azimuthal angle :math:`\phi`. xpcsjax operates on
   one :math:`q`-ring at a time and treats the :math:`\phi`-sectors of that
   ring as independent observations that share the underlying physics.

First-order correlation g_1
-----------------------------------


The normalised first-order correlation function (the field correlation)
measures the persistence of the scattered field:

.. math::
   :label: xb_g1

   g_1(\mathbf{q}, t_1, t_2)
   \;=\;
   \frac{\langle E^{*}(\mathbf{q}, t_1)\, E(\mathbf{q}, t_2)\rangle}
        {\sqrt{\langle I(\mathbf{q}, t_1)\rangle\,
               \langle I(\mathbf{q}, t_2)\rangle}}.

For displacement statistics that are Gaussian (which holds for large :math:`N`
by the central limit theorem) the field correlation factorises into an
internal (diffusive) and an external (advective) piece [He2024]_:

.. math::
   :label: xb_g1_factor

   g_1(\mathbf{q}, t_1, t_2)
   \;=\; \exp\!\left(-\tfrac{q^2}{2}\int_{t_1}^{t_2} J(t')\,dt'\right)
       \times
       \exp\!\left(i\,q\!\int_{t_1}^{t_2}\langle v(t')\rangle\,dt'\right).

The first factor is a generalised Debye--Waller decay controlled by the
transport coefficient :math:`J(t)` (see :doc:`transport_coefficient`); the
second is a phase shift from the mean particle drift.

Second-order correlation g_2 and c_2
----------------------------------------------------


The XPCS observable is the normalised intensity correlation function:

.. math::
   :label: xb_g2

   g_2(\mathbf{q}, t_1, t_2)
   \;=\;
   \frac{\langle I(\mathbf{q}, t_1)\, I(\mathbf{q}, t_2)\rangle}
        {\langle I(\mathbf{q}, t_1)\rangle\,
         \langle I(\mathbf{q}, t_2)\rangle}.

Throughout this documentation we follow the homodyne/heterodyne convention and
refer to the two-time intensity correlation as :math:`c_2(\mathbf{q}, t_1, t_2)`,
reserving :math:`g_2(q, \tau)` for the equilibrium (lag-only) projection.

By Wick's theorem applied to the Gaussian-field assumption, :math:`c_2` and
:math:`g_1` are linked by the **Siegert relation**:

.. math::
   :label: xb_siegert

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \beta(t_1, t_2)\,\bigl|g_1(\mathbf{q}, t_1, t_2)\bigr|^2,

where :math:`\beta \in (0, 1]` is the **speckle contrast** (the coherence
factor at the detector). For a perfectly coherent beam with single-mode
detection :math:`\beta = 1`; in realistic synchrotron XPCS,
:math:`\beta \approx 0.05`--:math:`0.8`.

.. note::

   Because :math:`c_2` depends on :math:`|g_1|^2`, the advective phase factor in
   Equation :eq:`xb_g1_factor` cancels for homodyne detection. It survives in
   heterodyne detection through cross terms between distinguishable scattering
   populations, which is the source of the characteristic oscillatory fringes
   used to extract velocities (:doc:`heterodyne_model`).

Two-time matrix
---------------

In a non-stationary experiment (yielding, aging, transient flow), :math:`c_2`
depends on the **two absolute times** :math:`t_1` and :math:`t_2`, not only on
the lag :math:`\tau = t_2 - t_1`. The natural representation is therefore a
matrix indexed by discrete frame times:

.. math::

   c_2^{ij} \;=\; c_2(q, t_i, t_j),
   \qquad i, j \in \{1, \dots, N_t\}.

This matrix is symmetric (:math:`c_2^{ij} = c_2^{ji}`) and has
:math:`c_2^{ii} = 1 + \beta` on the diagonal (zero lag). The standard
equilibrium lag-only correlation is recovered by averaging along each
anti-diagonal at lag :math:`\tau = (j - i)\,\Delta t`.

.. warning::

   Reducing the two-time matrix to :math:`g_2(q, \tau)` is appropriate only
   for stationary samples. For yielding suspensions or other transient
   processes the diagonal average mixes physically distinct regimes and biases
   any single-exponential or transport-coefficient fit. xpcsjax fits the full
   matrix.

Equilibrium projection g_2(q, tau)
-------------------------------------------


When the sample is genuinely stationary, the two-time correlation collapses to
a function of the lag only,

.. math::
   :label: xb_g2_equilibrium

   g_2(q, \tau)
   \;=\; \frac{\langle I(q, t)\, I(q, t + \tau)\rangle}{\langle I(q, t)\rangle^2}
   \;=\; 1 + \beta\, e^{-2\Gamma \tau},

where for Brownian diffusion :math:`\Gamma = D q^2`. This is the standard
single-exponential model familiar from dynamic light scattering. xpcsjax can
fit equilibrium data either with the dedicated static modes
(``static``, ``static_isotropic``, ``static_anisotropic``) or, equivalently,
through the laminar-flow kernel with zero shear rate; see
:doc:`homodyne_model`.

Analysis modes implemented in xpcsjax
-------------------------------------

xpcsjax exposes the following analysis modes through
:class:`xpcsjax.core.HomodyneModel` and :class:`xpcsjax.core.HeterodyneModel`:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Mode
     - Model class
     - Physical regime
   * - ``static_isotropic``
     - HomodyneModel
     - Equilibrium sample, :math:`g_1` averaged over :math:`\phi`.
   * - ``static_anisotropic``
     - HomodyneModel
     - Equilibrium sample, angle-resolved :math:`g_1`.
   * - ``laminar_flow``
     - HomodyneModel
     - Sheared suspension; adds the sinc\ :sup:`2` modulation from gap
       integration.
   * - ``two_component``
     - HeterodyneModel
     - Mixed-population sample (e.g. flowing band + non-flowing band); 14
       physics parameters plus 2 per-angle scaling parameters.

The per-mode parameter sets and kernel equations are derived in
:doc:`homodyne_model` and :doc:`heterodyne_model`.

Data path into xpcsjax
----------------------

The public loader :func:`xpcsjax.data.xpcs_loader.load_xpcs_data` returns a container whose
core array is ``c2_exp`` of shape ``(n_phi, n_time, n_time)``. The companion
arrays describe the experimental geometry:

* ``phi_angles``  --- azimuthal angles of the :math:`q`-ring sectors
  (radians, shape ``(n_phi,)``);
* ``t_lab``       --- laboratory time grid (seconds);
* ``q``           --- magnitude of the scattering vector for the selected
  ring (\ :math:`\text{\AA}^{-1}`).

The :class:`xpcsjax.core.HomodyneModel` and
:class:`xpcsjax.core.HeterodyneModel` classes provide the forward map
:math:`\theta \mapsto c_2^{\mathrm{model}}` that the NLSQ engine compares
against ``c2_exp``. For homodyne the forward call is:

.. code-block:: python

   from xpcsjax import HomodyneModel
   import jax.numpy as jnp

   model = HomodyneModel(mode="laminar_flow", q=q, h=h_gap)
   c2_model = model.compute_c2(
       params=jnp.array([D0, alpha, D_offset,
                         gamma_dot_0, beta_gamma, gamma_dot_offset, phi_0]),
       phi_angles=phi_angles,
       contrast=0.5,
       offset=1.0,
   )  # shape (n_phi, n_time, n_time)

For heterodyne the multi-angle dispatch returns one
:class:`xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult` per :math:`\phi` angle;
see :doc:`heterodyne_model`.

Fit objective
-------------

Given measured :math:`\{c_2^{ij,\,\mathrm{meas}}\}` and a forward model
:math:`c_2^{\mathrm{model}}(\theta)`, the NLSQ engine minimises a weighted
sum of squared residuals,

.. math::

   \chi^2(\theta)
   \;=\; \sum_{k=1}^{N_\phi}\sum_{i, j}
     w_{kij}\!\left[c_2^{kij,\,\mathrm{meas}}
                    - c_2^{kij,\,\mathrm{model}}(\theta)\right]^2,

via the trust-region Levenberg--Marquardt solver in the upstream NLSQ
library. xpcsjax owns the strategy: parameter transforms, anti-degeneracy
defence, multistart, and CMA-ES escape; NLSQ owns the trust-region step
itself. The weights :math:`w_{kij}` default to uniform; they can be specialised
to encode Poisson photon statistics or shear-sensitivity reweighting (see
:doc:`anti_degeneracy`).

Why two-time, why JAX, why float64
----------------------------------

Three implementation choices follow directly from the physics above:

1. **Two-time matrix.** Yielding and aging samples are non-stationary; the
   :math:`g_2(q, \tau)` projection cannot recover :math:`J(t)` correctly. The
   :math:`c_2` matrix preserves the full :math:`(t_1, t_2)` dependence.
2. **JAX.** The forward map evaluates an exponential of a cumulative
   trapezoidal integral over the time grid, then broadcasts across
   :math:`(n_\phi, n_t, n_t)`. JIT compilation and ``vmap`` over angles map
   directly onto this structure.
3. **Float64.** Physical parameters such as :math:`D_0` and
   :math:`\dot{\gamma}_0` span six or more orders of magnitude, and
   :math:`q^2 \mathcal{D}` can be small while :math:`q^2 \mathcal{D}` for
   neighbouring lags differ in the last few significant digits. xpcsjax
   enforces ``JAX_ENABLE_X64=1`` at import time; see
   :mod:`xpcsjax` for the environment setup.

.. seealso::

   * :doc:`correlation_functions` -- formal derivation of :math:`g_1`,
     :math:`g_2`, and the Siegert relation.
   * :doc:`homodyne_model` -- per-mode kernels for static and laminar-flow
     analyses.
   * :doc:`heterodyne_model` -- the two-component (mixed-population) kernel.
   * :doc:`transport_coefficient` -- definition and role of :math:`J(t)`.
   * :doc:`anti_degeneracy` -- five-layer defence that keeps the NLSQ fit
     identifiable across angles.
   * :doc:`citations` -- primary literature, including [He2024]_ and
     [He2025]_.
