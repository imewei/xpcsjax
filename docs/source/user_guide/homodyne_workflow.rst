Homodyne workflow
=================

.. currentmodule:: xpcsjax


This page walks through the homodyne pipeline end to end, first using
the two-function public API and then using :class:`xpcsjax.core.HomodyneModel`
directly for evaluating the model at known parameters.

The two-function path
---------------------

For any of the homodyne analysis modes (``static``,
``static_isotropic``, ``static_anisotropic``, ``laminar_flow``), the
canonical script is exactly two calls:

.. code-block:: python

   import xpcsjax

   data = xpcsjax.load_xpcs_data("homodyne_config.yaml")
   result = xpcsjax.fit_nlsq(data, "homodyne_config.yaml")

   if result.success:
       for name, value, sigma in zip(
           ["D0", "alpha", "D_offset"],
           result.parameters,
           result.uncertainties,
       ):
           print(f"{name:10s} = {value:.4e} ± {sigma:.2e}")
       print(f"reduced chi2 = {result.reduced_chi_squared:.3f}")
   else:
       print(f"fit did not converge: {result.message}")

The return value is an :class:`xpcsjax.optimization.nlsq.results.OptimizationResult`. Its full
field list is documented in :doc:`/user_guide/interpreting_results`.

A minimum-viable YAML for a static isotropic fit looks like:

.. code-block:: yaml

   analysis_mode: static_isotropic

   experimental_data:
     data_file_name: dataset.h5

   analyzer_parameters:
     temporal:
       dt: 0.05
       start_frame: 0
       end_frame: 2000
     scattering:
       wavevector_q: 0.012
     geometry:
       stator_rotor_gap: 1.0e-3   # ignored by static modes

   initial_parameters:
     values: [1.0e3, 0.0, 0.0]

   parameter_bounds:
     D0:       [1.0e1, 1.0e5]
     alpha:    [-0.5, 0.5]
     D_offset: [-1.0e3, 1.0e3]

   optimization:
     nlsq:
       max_iterations: 1000
       tolerance: 1.0e-8

Saving and post-processing
--------------------------

The result object can be inspected directly and serialised to disk:

.. code-block:: python

   print(result.parameters)         # 1-D array, length = active params
   print(result.uncertainties)      # same length, 1-sigma
   print(result.covariance.shape)   # (n_params, n_params)
   print(result.iterations)
   print(result.execution_time)     # wall seconds
   print(result.device_info)        # populated by xpcsjax.device
   print(result.quality_flag)       # 'good' | 'warn' | 'bad'

The ``nlsq_diagnostics``, ``streaming_diagnostics``, and
``stratification_diagnostics`` fields are structured dictionaries that
record decisions made by the strategy selector, the anti-degeneracy
controller, and the data-prep stage. They are intended to be machine-
readable; see :doc:`/user_guide/interpreting_results` for the
field-by-field description.

Using :class:`xpcsjax.core.HomodyneModel` directly
---------------------------------------------

You sometimes need the forward model without going through the
optimiser — for example, to plot what the configured initial
parameters predict before launching a fit, or to compute a residual
against a known parameter set. :class:`xpcsjax.core.HomodyneModel` exposes
the forward kernel.

Construction
~~~~~~~~~~~~

The model takes a single ``config`` ``dict``:

.. code-block:: python

   from xpcsjax import HomodyneModel

   model_config = {
       "analyzer_parameters": {
           "temporal":   {"dt": 0.05, "start_frame": 0, "end_frame": 2000},
           "scattering": {"wavevector_q": 0.012},
           "geometry":   {"stator_rotor_gap": 1.0e-3},
       },
       "analysis_mode": "static_isotropic",
   }
   model = HomodyneModel(model_config)

After construction, the model exposes:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``time_array``
     - 1-D array of times derived from ``dt`` and the frame range.
   * - ``t1_grid``, ``t2_grid``
     - The 2-D index grids used to evaluate :math:`g_2(t_1, t_2)`.
   * - ``dt``
     - The frame-to-frame time step (seconds).
   * - ``wavevector_q``
     - The configured :math:`q` value (inverse Ångström).
   * - ``stator_rotor_gap``
     - The geometric gap, relevant in laminar-flow analyses.
   * - ``analysis_mode``
     - Echo of the configured mode.
   * - ``model``
     - Internal handle to the model registry entry.
   * - ``physics_factors``
     - Cached prefactors (avoids recomputation across calls).

Evaluating the model
~~~~~~~~~~~~~~~~~~~~

The two evaluation methods are :meth:`xpcsjax.core.HomodyneModel.compute_c2` and
:meth:`xpcsjax.core.HomodyneModel.compute_c2_single_angle`.

.. code-block:: python

   import jax.numpy as jnp

   params = jnp.array([1.0e3, 0.0, 0.0])              # static_isotropic
   phi_angles = jnp.array([0.0, 30.0, 60.0, 90.0])    # degrees

   c2 = model.compute_c2(
       params,
       phi_angles,
       contrast=0.5,
       offset=1.0,
   )
   # c2.shape == (n_phi, n_time, n_time)
   #          == (4, n_time, n_time)

For a single angle there is a faster path that returns one 2-D
correlation matrix:

.. code-block:: python

   c2_single = model.compute_c2_single_angle(
       params,
       phi=0.0,
       contrast=0.5,
       offset=1.0,
   )
   # c2_single.shape == (n_time, n_time)

Both methods are JIT-compiled on first call and float64 throughout.
The returned tensor is a ``jax.numpy.ndarray``; convert with
``np.asarray(...)`` if you need NumPy for plotting.

A complete script
-----------------

.. code-block:: python

   import xpcsjax
   import numpy as np
   import matplotlib.pyplot as plt

   data = xpcsjax.load_xpcs_data("homodyne_config.yaml")
   result = xpcsjax.fit_nlsq(data, "homodyne_config.yaml")
   print(f"converged: {result.success}, "
         f"reduced chi2: {result.reduced_chi_squared:.3f}")

   # Re-build the forward model so we can plot the fitted curve.
   from xpcsjax import HomodyneModel, ConfigManager
   cfg = ConfigManager("homodyne_config.yaml")
   cfg.load_config()

   model = HomodyneModel(cfg.config)
   phi = data["phi_angles_list"]
   c2_model = np.asarray(model.compute_c2(result.parameters, phi))

   c2_exp = np.asarray(data["c2_exp"])
   fig, ax = plt.subplots(1, 2, figsize=(10, 4))
   ax[0].imshow(c2_exp[0]);    ax[0].set_title("experimental")
   ax[1].imshow(c2_model[0]);  ax[1].set_title("model fit")
   plt.show()

Performance notes
-----------------

* The first call to :meth:`xpcsjax.core.HomodyneModel.compute_c2` triggers JIT compilation. On a
  modern CPU this typically completes in a few seconds for the
  three-parameter static models and somewhat longer for laminar flow.
  Subsequent calls reuse the cached compilation.
* The contrast and offset arguments default to ``contrast=0.5`` and
  ``offset=1.0``. They are not free parameters in homodyne mode — they
  exist so that ``compute_c2`` can be re-used for plotting against the
  experimental data which has already been Siegert-transformed.
* For static modes the ``laminar_flow``-specific shear weighting is
  ignored; do not interpret a small ``stator_rotor_gap`` value as
  affecting a static fit.

Next: :doc:`/user_guide/heterodyne_workflow` covers the
two-component pipeline, which has a different return type and a
distinct configuration layout.
