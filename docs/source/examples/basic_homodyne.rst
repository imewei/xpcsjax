Basic Homodyne Fit (Static Isotropic)
=====================================

.. currentmodule:: xpcsjax


This page works through a minimal homodyne XPCS fit in the
``static_isotropic`` analysis mode, which carries only three free
parameters: ``D0``, ``alpha``, and ``D_offset``. The goal is to exercise
the public API end-to-end on a small problem and show how to read the
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult` that
comes back.

Configuration
-------------

xpcsjax is configuration-driven. The same YAML document is consumed by
:func:`~xpcsjax.data.xpcs_loader.load_xpcs_data`,
:class:`~xpcsjax.config.ConfigManager`, and
:func:`~xpcsjax.optimization.nlsq.fit_nlsq`. A minimal static-isotropic config looks like
this:

.. code-block:: yaml

    # config_static_isotropic.yaml
    analysis_settings:
      analysis_mode: static_isotropic
      static_submode: isotropic

    experimental_data:
      data_folder_path: ./data/
      data_file_name: example_c2.npz
      phi_angles_path: ./data/
      phi_angles_file: phi_angles.txt

    analyzer_parameters:
      temporal:
        dt: 0.1
        start_frame: 1
        end_frame: 401
      scattering:
        wavevector_q: 0.0054
      geometry:
        stator_rotor_gap: 2.0e6

    initial_parameters:
      values: [1.0e3, -1.5, 1.0e2]
      parameter_names: [D0, alpha, D_offset]

    parameter_space:
      bounds:
        - {name: D0,        min: 1.0,    max: 1.0e6}
        - {name: alpha,     min: -2.0,   max: 2.0}
        - {name: D_offset,  min: 0.0,    max: 1.0e4}

    optimization_config:
      angle_filtering:
        enabled: false

.. note::

   In ``static_isotropic`` mode the per-angle dependence collapses, so
   the fit reduces to a single isotropic diffusion law parameterised by
   the three coefficients listed above.

Running the fit
---------------

The public entry points are lazily loaded; importing ``xpcsjax`` itself
is cheap because JAX is not pulled in until one of the six exported
names is first accessed.

.. code-block:: python

    from pathlib import Path

    from xpcsjax import ConfigManager, fit_nlsq, load_xpcs_data

    config_path = Path("config_static_isotropic.yaml")

    data = load_xpcs_data(str(config_path))
    print(sorted(data.keys()))
    # ['c2_exp', 'phi_angles_list', 't1', 't2', 'wavevector_q_list', ...]

    result = fit_nlsq(data, str(config_path))

The :func:`~xpcsjax.optimization.nlsq.fit_nlsq` call returns an
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult` for
homodyne modes (heterodyne returns ``list[NLSQResult]``; see
:doc:`heterodyne_multiangle`).

Inspecting the result
---------------------

Every field on the result object is populated by the NLSQ engine and
post-processed by the xpcsjax result builder. The most common fields
are summarised below.

.. code-block:: python

    print(result.success)             # bool, from .convergence_status
    print(result.message)             # short human-readable status
    print(result.parameters)          # jnp.ndarray, shape (n_params,)
    print(result.uncertainties)       # jnp.ndarray, shape (n_params,)
    print(result.covariance.shape)    # (n_params, n_params)
    print(result.reduced_chi_squared)
    print(result.iterations)
    print(result.execution_time)      # seconds
    print(result.device_info)         # JAX device used
    print(result.quality_flag)        # data-quality verdict
    print(result.sigma_is_default)    # True if no per-point sigma was supplied

The recovery audit trail (see :doc:`/advanced/anti_degeneracy`) is
exposed through ``result.recovery_actions``. The memory-routing audit
appears in ``result.streaming_diagnostics`` and
``result.stratification_diagnostics``, and the raw upstream NLSQ
metadata in ``result.nlsq_diagnostics``.

.. important::

   When ``sigma_is_default`` is ``True`` the reported
   ``reduced_chi_squared`` is on the same arbitrary scale as the
   identity weighting. Do not interpret it as an absolute goodness-of-fit
   in that case; instead trust the per-parameter uncertainties.

Plotting the fit
----------------

For a quick visual check, evaluate the
:class:`~xpcsjax.core.HomodyneModel` at the best-fit parameters and
overlay it on the data at one ``phi`` slice.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    from xpcsjax import HomodyneModel

    cm = ConfigManager(str(config_path))
    cm.load_config()
    model_cfg = cm.get_model()
    model = HomodyneModel(model_cfg)

    phi = np.asarray(data["phi_angles_list"])
    c2_exp = np.asarray(data["c2_exp"])

    c2_pred = model.compute_c2(
        result.parameters,
        phi,
        contrast=0.5,
        offset=1.0,
    )

    idx = 0  # first angle
    fig, ax = plt.subplots()
    ax.imshow(np.asarray(c2_pred[idx]), origin="lower")
    ax.set_title(f"Predicted c2, phi={float(phi[idx]):.2f}")
    fig.savefig("c2_fit_static_isotropic.png", dpi=150)

The full ``c2_pred`` tensor has shape ``(n_phi, n_time, n_time)``.

Next steps
----------

- :doc:`laminar_flow` — switch to a flow-aware mode with seven free
  parameters.
- :doc:`multistart_robust_fit` — combat local minima with
  :func:`~xpcsjax.optimization.nlsq.core.fit_nlsq_multistart`.
- :doc:`/advanced/architecture` — understand what
  :func:`~xpcsjax.optimization.nlsq.fit_nlsq` is doing under the hood.
