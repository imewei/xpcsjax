Heterodyne Multi-angle Fit (two_component mode)
===============================================

.. currentmodule:: xpcsjax


Heterodyne XPCS fits resolve a two-component model (reference + sample)
with shared physics but per-angle scaling factors. Unlike homodyne,
:func:`~xpcsjax.optimization.nlsq.fit_nlsq` returns a ``list[NLSQResult]`` — one element
per angle — so iteration over the list is the standard pattern.

.. important::

   The public :class:`~xpcsjax.core.HeterodyneModel` symbol is still gated
   by an ``xfail`` marker at the lazy-API boundary (see
   :doc:`/advanced/lazy_api`). Construction through
   ``HeterodyneModel.from_config(yaml_dict)`` in
   :mod:`xpcsjax.core.heterodyne_model_stateful` is the supported path
   for production fits until that gate is lifted in Phase 6.

Configuration
-------------

The crucial detail in a heterodyne YAML is the
``optimization.nlsq`` block. Heterodyne nests NLSQ tuning options one
level deeper than homodyne; xpcsjax unwraps this nesting before
handing the dictionary to the adapter, so omitting the ``nlsq:`` key
silently turns off optimizer-level overrides.

.. code-block:: yaml

    # config_heterodyne.yaml
    analysis_settings:
      analysis_mode: two_component
      heterodyne_submode: full

    experimental_data:
      data_folder_path: ./data/
      data_file_name: example_c2_het.npz
      phi_angles_path: ./data/
      phi_angles_file: phi_angles.txt

    analyzer_parameters:
      temporal:
        dt: 0.05
        start_frame: 1
        end_frame: 801
      scattering:
        wavevector_q: 0.0072
      geometry:
        stator_rotor_gap: 2.0e6

    initial_parameters:
      values:
        [1.0e3, -1.5, 1.0e2,
         1.0e-3, 0.0, 0.0,
         0.0,
         0.05, 1.05]
      parameter_names:
        [D0, alpha, D_offset,
         gamma_dot_t0, beta, gamma_dot_t_offset,
         phi0,
         contrast, offset]

    parameter_space:
      bounds:
        - {name: D0,                  min: 1.0,    max: 1.0e6}
        - {name: alpha,               min: -2.0,   max: 2.0}
        - {name: D_offset,            min: 0.0,    max: 1.0e4}
        - {name: gamma_dot_t0,        min: 0.0,    max: 1.0}
        - {name: beta,                min: -2.0,   max: 2.0}
        - {name: gamma_dot_t_offset,  min: 0.0,    max: 1.0}
        - {name: phi0,                min: -90.0,  max: 90.0}
        - {name: contrast,            min: 0.0,    max: 1.0}
        - {name: offset,              min: 0.5,    max: 1.5}

    optimization:
      nlsq:
        max_nfev: 2000
        ftol: 1.0e-10
        xtol: 1.0e-10
        gtol: 1.0e-10
        cmaes_escape:
          enabled: true

    anti_degeneracy:
      enabled: true
      fourier_reparam:
        enabled: true
      shear_weighting:
        enabled: true

.. warning::

   ``optimization.nlsq`` is **not** the same as the top-level
   ``nlsq:`` key used by homodyne. Keep the ``optimization:`` parent
   in heterodyne configs — the heterodyne adapter looks there first.

Running the fit
---------------

.. code-block:: python

    from pathlib import Path

    from xpcsjax import ConfigManager, fit_nlsq, load_xpcs_data

    config_path = Path("config_heterodyne.yaml")

    data = load_xpcs_data(str(config_path))
    results = fit_nlsq(data, str(config_path))
    # heterodyne path → list of per-angle NLSQResult

    print(type(results).__name__, len(results))
    # list 12   (for a 12-angle dataset, say)

Iterating per angle
-------------------

Each entry is an
:class:`~xpcsjax.optimization.nlsq.heterodyne_results.NLSQResult`
holding the per-angle best parameters, uncertainties, and diagnostics.

.. code-block:: python

    import numpy as np

    phi = np.asarray(data["phi_angles_list"])

    for angle_deg, r in zip(phi, results, strict=True):
        if not r.success:
            print(f"phi={float(angle_deg):7.2f}: FAILED ({r.message})")
            continue
        print(
            f"phi={float(angle_deg):7.2f}  "
            f"chi2_red={float(r.reduced_chi_squared): .4e}  "
            f"iters={int(r.iterations):4d}  "
            f"t={float(r.execution_time): .2f}s"
        )

Cross-angle parameter consistency is the usual sanity check: shared
physics parameters (e.g. ``D0``, ``alpha``) should agree within their
reported uncertainties across angles, while per-angle scaling
parameters (e.g. ``contrast``, ``offset``) are allowed to vary.

.. code-block:: python

    names = ["D0", "alpha", "D_offset"]
    for k, name in enumerate(names):
        vals = np.array([float(r.parameters[k]) for r in results if r.success])
        print(f"{name:>10s}  mean={vals.mean(): .4e}  std={vals.std(): .2e}")

Fourier reparameterisation in the multi-angle setting
-----------------------------------------------------

With many angles, the shear sub-space picks up additional null
directions: angles near ``phi0`` are insensitive, angles in the
flow-perpendicular direction are dominant. The
:class:`~xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer`
operates in a Fourier basis indexed by harmonics of ``2 * phi``, which
absorbs that anisotropy cleanly. You will see the controller log
entries on each angle's ``recovery_actions`` field.

Next steps
----------

- :doc:`multistart_robust_fit` — combine multi-angle heterodyne with
  multi-start sampling for robustness.
- :doc:`/advanced/cma_es_escape` — CMA-ES is on by default for
  heterodyne fits; this page explains why.
- :doc:`/advanced/anti_degeneracy` — controller layers and their
  cost.
