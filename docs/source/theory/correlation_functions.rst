.. _theory_correlation_functions:

Correlation Functions in XPCS
=============================

This page derives the two-time intensity correlation function
:math:`c_2(\mathbf{q}, t_1, t_2)` from first principles, states the Siegert
relation, and explains why the equilibrium projection :math:`g_2(q, \tau)`
is insufficient for the non-stationary samples that xpcsjax targets. The
derivation follows the transport-coefficient framework of [He2024]_.

Position density and scattered field
------------------------------------

For a sample of :math:`N` scattering centres at positions
:math:`\mathbf{r}_j(t)`, the position density in Fourier space is

.. math::
   :label: cf_rho

   \rho(\mathbf{q}, t) \;=\; \sum_{j=1}^N f_j
       \exp\!\left(i\,\mathbf{q}\cdot\mathbf{r}_j(t)\right),

where :math:`f_j` is the form factor of particle :math:`j` and the magnitude
of the momentum transfer is :math:`q = 4\pi \sin(\theta)/\lambda`.

The scattered electric field is

.. math::
   :label: cf_E

   E(\mathbf{q}, t) \;\propto\; \rho(\mathbf{q}, t),

and the measured intensity per pixel is

.. math::
   :label: cf_I

   I(\mathbf{q}, t) \;=\; |E(\mathbf{q}, t)|^2
   \;=\; \left|\sum_j f_j\, e^{i\mathbf{q}\cdot \mathbf{r}_j(t)}\right|^2.

The fluctuations of :math:`I(\mathbf{q}, t)` carry the information about the
sample's collective dynamics.

First-order correlation function c_1
--------------------------------------------


The normalised first-order correlation function is

.. math::
   :label: cf_c1

   c_1(\mathbf{q}, t_1, t_2)
   \;=\;
   \frac{\langle E^{*}(\mathbf{q}, t_1)\, E(\mathbf{q}, t_2)\rangle}
        {\sqrt{\langle I(\mathbf{q}, t_1)\rangle\,
               \langle I(\mathbf{q}, t_2)\rangle}}.

For a Gaussian process --- valid for large :math:`N` by the central limit
theorem --- :math:`c_1` depends only on the **transport coefficient**
:math:`J(t)` and the mean particle velocity:

.. math::
   :label: cf_c1_general

   c_1(\mathbf{q}, t_1, t_2)
   \;=\;
   \exp\!\left(-\tfrac{q^2}{2}\int_{t_1}^{t_2} J(t')\,dt'\right)
   \;\times\;
   \exp\!\left(i\,q\!\int_{t_1}^{t_2}\langle v(t')\rangle\,dt'\right).

The first factor is a generalised Debye--Waller decay; the second is a phase
shift from the mean drift. This factorisation is exact for Gaussian
displacement statistics. We split

.. math::
   :label: cf_factorisation

   c_1 \;=\; c_1^{(\mathrm{in})} \;\times\; c_1^{(\mathrm{ex})},

with the internal (diffusive) piece

.. math::

   c_1^{(\mathrm{in})}(\mathbf{q}, t_1, t_2)
   \;=\;
   \exp\!\left(-\tfrac{q^2}{2}\,\mathcal{D}(t_1, t_2)\right),
   \qquad
   \mathcal{D}(t_1, t_2) \;=\; \int_{t_1}^{t_2} J(t')\,dt',

and the external (drift) piece

.. math::

   c_1^{(\mathrm{ex})}(\mathbf{q}, t_1, t_2)
   \;=\;
   \exp\!\left(i\,q\!\int_{t_1}^{t_2}\langle v(t')\rangle\,dt'\right).

The transport coefficient :math:`J(t)` is the central physical quantity in
xpcsjax; it is defined in :doc:`transport_coefficient`.

Second-order (intensity) correlation function c_2
---------------------------------------------------------


The normalised second-order correlation function is

.. math::
   :label: cf_c2

   c_2(\mathbf{q}, t_1, t_2)
   \;=\;
   \frac{\langle I(\mathbf{q}, t_1)\, I(\mathbf{q}, t_2)\rangle}
        {\langle I(\mathbf{q}, t_1)\rangle\,
         \langle I(\mathbf{q}, t_2)\rangle}.

This is the quantity measured directly in XPCS experiments. Each detector
pixel contributes one time series :math:`I(\mathbf{q}, t)` and the correlation
is accumulated as a function of the absolute times :math:`t_1` and
:math:`t_2`.

.. note::

   :math:`c_2` is dimensionless and satisfies :math:`c_2 \geq 1` for classical
   intensity fluctuations (Cauchy--Schwarz). For ergodic decorrelating
   dynamics, :math:`c_2(t_1, t_2) \to 1` as :math:`|t_2 - t_1| \to \infty`.

The Siegert relation
--------------------

For a Gaussian field (single-mode thermal-light statistics), Wick's theorem
yields the **Siegert relation** connecting the measured intensity correlation
to the unmeasurable field correlation [Sutton2008]_:

.. math::
   :label: cf_siegert

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \beta(t_1, t_2)\,
         \bigl|c_1(\mathbf{q}, t_1, t_2)\bigr|^2,

where :math:`\beta \in (0, 1]` is the **speckle contrast** (also called the
optical coherence parameter). For a fully coherent beam with single-mode
detection :math:`\beta = 1`; in realistic experiments
:math:`\beta \approx 0.05`--:math:`0.8` depending on beam coherence, pixel
geometry, and sample inhomogeneity.

Substituting :eq:`cf_c1_general` into :eq:`cf_siegert` gives the general
homodyne result:

.. math::
   :label: cf_c2_homodyne

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \beta(t_1, t_2)\,
         \exp\!\left(-q^2\!\int_{t_1}^{t_2} J(t')\,dt'\right).

The drift phase drops out of :math:`|c_1|^2` for homodyne detection. It is
recovered in heterodyne detection (multi-component scattering) through cross
terms between the populations, leading to the characteristic oscillations
that xpcsjax fits in ``two_component`` mode (:doc:`heterodyne_model`).

.. note::

   In xpcsjax, the diffusion integral
   :math:`\mathcal{D}(t_1, t_2) = \int_{t_1}^{t_2} J(t')\,dt'` is evaluated
   numerically by cumulative trapezoidal integration on the experimental time
   grid --- no closed-form antiderivative is ever substituted, because the
   power-law parameterisation :math:`J(t) = D_0 t^\alpha + D_\mathrm{offset}`
   does not admit one for generic :math:`\alpha`. See
   :doc:`transport_coefficient` for the integration convention and the
   :math:`D_0 = 2 D_\mathrm{SE}` factor.

Wiener--Khinchin and the equilibrium projection
-----------------------------------------------

For a wide-sense stationary process, the Wiener--Khinchin theorem relates the
autocorrelation function to the power spectral density:
the second-moment statistics of :math:`I(\mathbf{q}, t)` are completely
specified by the lag-only correlation. Equivalently, the two-time matrix
collapses onto a single curve in :math:`\tau = t_2 - t_1`:

.. math::
   :label: cf_g2_equilibrium

   g_2(q, \tau)
   \;=\;
   \frac{\langle I(q, t)\, I(q, t + \tau)\rangle}{\langle I(q, t)\rangle^2}
   \;=\; 1 + \beta\, e^{-2\Gamma \tau}.

For simple Brownian diffusion :math:`\Gamma = D q^2` and the decay is a
single exponential. The :math:`g_2(q, \tau)` form is widely used in dynamic
light scattering and equilibrium XPCS analysis.

.. warning::

   Applying :math:`g_2(q, \tau)` analysis to a non-stationary sample produces
   artefact-corrupted parameters: the effective :math:`D` absorbs
   time-averaged heterogeneity, the apparent contrast :math:`\beta` is
   depressed, and the functional form may stop being a single exponential
   even when the underlying physics is simple. For yielding, aging, or
   shear-banding samples, use the full two-time matrix.

The two-time correlation matrix
-------------------------------

In practice :math:`c_2` is represented as a matrix indexed by discrete frame
times:

.. math::

   c_2^{ij} \;=\; c_2(q, t_i, t_j),
   \qquad i, j \in \{1, \dots, N_t\}.

This matrix is symmetric (:math:`c_2^{ij} = c_2^{ji}`) and has
:math:`c_2^{ii} = 1 + \beta` on the diagonal (zero lag). The equilibrium
projection :math:`g_2(\tau)` corresponds to the mean along the anti-diagonal
at lag :math:`\tau = (j - i)\,\Delta t`.

The xpcsjax data loader :func:`xpcsjax.data.xpcs_loader.load_xpcs_data` returns the array
``c2_exp`` of shape ``(n_phi, n_time, n_time)`` together with the laboratory
time grid ``t_lab`` and the angle vector ``phi_angles``. The angle axis is
preserved because the homodyne laminar-flow kernel and the heterodyne
two-component kernel both depend explicitly on :math:`\phi`.

Multi-angle data and the per-angle scaling
------------------------------------------

A typical XPCS experiment captures the full :math:`(q, \phi)` plane on a 2D
detector and resolves it into :math:`N_\phi` azimuthal sectors at each
:math:`q`. Each sector is fit simultaneously, but the speckle contrast
:math:`\beta(\phi)` and the baseline offset
:math:`c_\mathrm{offset}(\phi)` vary with angle for purely instrumental
reasons (partial coherence, pixel geometry, gain variation, sample
inhomogeneity).

The xpcsjax forward model factorises this as

.. math::
   :label: cf_per_angle

   c_2^{\mathrm{model}}(\phi_k, t_1, t_2; \theta)
   \;=\; c_\mathrm{offset}(\phi_k)
   \;+\; \beta(\phi_k)\,
       \left|c_1(\mathbf{q}, t_1, t_2; \theta)\right|^2,

where :math:`\theta` collects the physical parameters
(:math:`D_0, \alpha, D_\mathrm{offset}`, plus shear parameters in
``laminar_flow``) and :math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))` are
the per-angle scaling parameters. The naive choice of
:math:`2 N_\phi` free scaling parameters introduces a parameter
absorption degeneracy; xpcsjax breaks the degeneracy through the strategies
described in :doc:`anti_degeneracy`.

Fitting the model to data
-------------------------

Given measured :math:`\{c_2^{kij,\,\mathrm{meas}}\}` and the forward model
in Equation :eq:`cf_per_angle`, the NLSQ engine in xpcsjax minimises

.. math::

   \chi^2(\theta)
   \;=\; \sum_{k=1}^{N_\phi}\sum_{i, j}
     w_{kij}\!\left[c_2^{kij,\,\mathrm{meas}}
                    - c_2^{kij,\,\mathrm{model}}(\theta)\right]^2,

through the trust-region Levenberg--Marquardt step in the upstream NLSQ
library (:func:`xpcsjax.optimization.nlsq.fit_nlsq`). The weights :math:`w_{kij}` default to
uniform and can optionally encode Poisson photon statistics or
shear-sensitivity reweighting. The shape parameters
:math:`(\beta(\phi_k), c_\mathrm{offset}(\phi_k))` enter through the
anti-degeneracy controller's per-angle mode --- ``constant`` (fixed from
quantiles), ``auto`` (averaged and optimised), ``fourier`` (truncated
series), or ``individual``. See :doc:`anti_degeneracy`.

.. seealso::

   * :doc:`transport_coefficient` -- physical meaning of :math:`J(t)`.
   * :doc:`homodyne_model` -- mode-specific kernels for :math:`c_2`.
   * :doc:`heterodyne_model` -- multi-component generalisation.
   * :class:`xpcsjax.core.HomodyneModel` -- JAX implementation of the kernel.
   * :func:`xpcsjax.optimization.nlsq.fit_nlsq` -- the single-entry NLSQ wrapper.
   * :doc:`citations` -- bibliography.
