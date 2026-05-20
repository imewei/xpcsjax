Quickstart
==========

This page walks through the smallest end-to-end fit. It assumes you've
followed :doc:`installation`.

The whole API for a basic fit is two functions:

.. code-block:: python

   from xpcsjax import load_xpcs_data, fit_nlsq

   data   = load_xpcs_data("config.yaml")
   result = fit_nlsq(data, "config.yaml")

That's the same shape :func:`xpcsjax.data.xpcs_loader.load_xpcs_data` is documented with in its
own docstring, and :func:`xpcsjax.optimization.nlsq.fit_nlsq` is the single dispatch entry point
for both homodyne and heterodyne modes.

Step 1 — Configuration
----------------------

xpcsjax reads YAML configs that mirror the upstream homodyne / heterodyne
schema. A minimal homodyne config looks like:

.. code-block:: yaml

   # config.yaml
   analysis_mode: static_isotropic        # also: static_anisotropic | laminar_flow | two_component

   experimental_data:
     data_file_name: experiment.h5
     data_folder_path: /path/to/data

   analyzer_parameters:
     temporal:
       dt: 0.001                          # time step [s]
       start_frame: 0
       end_frame: 999
     scattering:
       wavevector_q: 0.0237               # [Å⁻¹]
     geometry:
       stator_rotor_gap: 2.0e6            # [Å]

   initial_parameters:
     active:
       - D0
       - alpha
       - D_offset
     values:
       D0: 1.0e-4
       alpha: 1.0
       D_offset: 0.0

   parameter_bounds:
     D0:        [1.0e-8, 1.0e-2]
     alpha:     [-2.0,   2.0]
     D_offset:  [-1.0e-4, 1.0e-4]

   optimization:
     nlsq:
       max_iterations: 200

See :doc:`user_guide/configuration` for the full schema including heterodyne
parameters (``two_component`` mode adds 14 physics parameters and 2 scaling
parameters).

Step 2 — Load data
------------------

.. code-block:: python

   >>> from xpcsjax import load_xpcs_data
   >>> data = load_xpcs_data("config.yaml")
   >>> sorted(data.keys())
   ['c2_exp', 'phi_angles_list', 't1', 't2', 'wavevector_q_list']

The loader returns a dictionary with five canonical keys:

* ``wavevector_q_list`` — 1-D array of q-values (Å⁻¹).
* ``phi_angles_list`` — 1-D array of detector φ angles (degrees).
* ``t1``, ``t2`` — time arrays for the correlation matrices.
* ``c2_exp`` — the experimental ``g2`` stack. Shape ``(n_phi, n_time, n_time)``
  for a multi-angle dataset, ``(n_time, n_time)`` for a single angle.

You can also pass a pre-built dict instead of a path:

.. code-block:: python

   data = load_xpcs_data({"data_file": "experiment.h5", "analysis_mode": "static_isotropic"})

or pass the dict via the explicit keyword:

.. code-block:: python

   data = load_xpcs_data(config_dict=cfg_dict)

These two-and-a-half-argument permutations match the actual loader signature
in :func:`xpcsjax.data.xpcs_loader.load_xpcs_data`.

Step 3 — Run the fit
--------------------

.. code-block:: python

   >>> from xpcsjax import fit_nlsq
   >>> result = fit_nlsq(data, "config.yaml")

The second argument can be a path *or* a pre-built
:class:`~xpcsjax.config.ConfigManager`. Passing the path is the common case
and re-uses the same YAML you used to load the data; the wrapper instantiates
``ConfigManager`` internally.

What you get back depends on ``analysis_mode``:

* **Homodyne** (``static``, ``static_isotropic``, ``laminar_flow``): a single
  :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`.
* **Heterodyne** (``two_component``): a list of ``NLSQResult`` objects, one
  per φ angle from the joint multi-angle fit.

This is the verbatim dispatch contract in :func:`xpcsjax.optimization.nlsq.fit_nlsq`.

Step 4 — Inspect the result
---------------------------

For the homodyne path:

.. code-block:: python

   >>> result.parameters             # np.ndarray, fitted parameters
   array([1.02e-04, 0.97, 5.3e-06])
   >>> result.uncertainties          # std-devs from covariance diagonal
   array([1.1e-06, 0.03, 4.2e-07])
   >>> result.chi_squared            # sum of squared residuals
   12345.6
   >>> result.reduced_chi_squared    # χ² / (n_data − n_params)
   1.02
   >>> result.convergence_status     # 'converged' | 'max_iter' | 'failed'
   'converged'
   >>> result.iterations
   47
   >>> result.success                # bool property
   True

The full field list — including diagnostics like ``device_info``,
``recovery_actions``, ``quality_flag``, ``streaming_diagnostics``,
``stratification_diagnostics``, ``nlsq_diagnostics``, and ``sigma_is_default``
— is in :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`.

Step 5 — What just happened?
----------------------------

The single :func:`~xpcsjax.optimization.nlsq.fit_nlsq` call ran the full xpcsjax NLSQ pipeline:

1. **ConfigManager** loaded and validated the YAML, normalised the analysis
   mode, and produced bounds + initial parameters from the registry.
2. **Strategy selection** — :func:`xpcsjax.optimization.nlsq.select_nlsq_strategy`
   consulted the dataset size, available RAM, and angle count to pick between
   the in-memory, stratified, streaming, or out-of-core paths.
3. **5-layer anti-degeneracy controller** wrapped the residual to prevent the
   well-known XPCS parameter degeneracies; see
   :doc:`advanced/anti_degeneracy`.
4. **NLSQ** ran the trust-region (Levenberg–Marquardt) solve via the JIT
   ``CurveFit`` cache. xpcsjax never calls NLSQ's higher-level ``fit()``
   wrapper.
5. **CMA-ES escape** triggered automatically if the trust-region solve
   plateaued above a threshold (heterodyne only by default).
6. **Result builder** packed parameters, covariance, χ², and diagnostics
   into :class:`~xpcsjax.optimization.nlsq.results.OptimizationResult` (or
   a per-angle list for heterodyne).

For deeper coverage of any one of those, follow the links above or read
:doc:`user_guide/nlsq_fitting`.

Next steps
----------

* :doc:`user_guide/homodyne_workflow` — full homodyne example with shear flow.
* :doc:`user_guide/heterodyne_workflow` — multi-angle heterodyne with Fourier
  reparameterisation.
* :doc:`examples/index` — worked end-to-end scripts.
* :doc:`api/public` — every public symbol with full signatures.
