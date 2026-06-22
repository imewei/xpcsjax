Heterodyne Multi-angle Fit (two_component mode)
===============================================

.. currentmodule:: xpcsjax


Heterodyne XPCS fits resolve a two-component model (reference + sample) with a
velocity (flow) term and a time-dependent mixing fraction. The 14 physics
parameters are shared across angles while per-angle scaling is fit jointly.
Like homodyne, :func:`~xpcsjax.optimization.nlsq.fit_nlsq` returns a **single**
:class:`~xpcsjax.optimization.nlsq.results.OptimizationResult`; the joint
multi-:math:`\phi` fit packs its per-angle detail (``chi2_per_angle``,
``parameter_names``, ``contrast_per_angle`` / ``offset_per_angle``) into
:attr:`result.nlsq_diagnostics <xpcsjax.optimization.nlsq.results.OptimizationResult.nlsq_diagnostics>`.

.. note::

   :class:`~xpcsjax.core.HeterodyneModel` is a fully public lazy export (Phase 6
   brought it to per-angle-mode parity with homodyne).
   ``HeterodyneModel.from_config(yaml_dict)`` in
   :mod:`xpcsjax.core.heterodyne_model_stateful` is the supported construction
   path for production fits.

Configuration
-------------

The crucial detail in a heterodyne YAML is the ``optimization.nlsq`` block.
Heterodyne nests NLSQ tuning options one level deeper than homodyne; xpcsjax
unwraps this nesting before handing the dictionary to the adapter, so omitting
the ``nlsq:`` key silently turns off optimizer-level overrides.

The 14 physics parameters use the registry's canonical heterodyne names. Note in
particular ``v_beta`` (the velocity exponent, **not** ``beta``) and ``phi0_het``
(the heterodyne flow angle in degrees, **not** ``phi0``) — both are renamed in
``xpcsjax/config/parameter_registry.py`` to disambiguate them from the homodyne
shear exponent ``beta`` and flow angle ``phi0``.

.. code-block:: yaml

    # config_heterodyne.yaml
    analysis_settings:
      analysis_mode: two_component

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

    initial_parameters:
      values:
        [1.0e4, 0.0, 0.0,
         1.0e4, 0.0, 0.0,
         1.0e3, 1.0, 0.0,
         0.5, 0.0, 0.0, 0.0,
         0.0,
         0.5, 1.0]
      parameter_names:
        [D0_ref, alpha_ref, D_offset_ref,
         D0_sample, alpha_sample, D_offset_sample,
         v0, v_beta, v_offset,
         f0, f1, f2, f3,
         phi0_het,
         contrast, offset]

    parameter_space:
      bounds:
        - {name: D0_ref,            min: 1.0,     max: 1.0e6}
        - {name: alpha_ref,         min: -2.0,    max: 2.0}
        - {name: D_offset_ref,      min: 0.0,     max: 1.0e4}
        - {name: D0_sample,         min: 1.0,     max: 1.0e6}
        - {name: alpha_sample,      min: -2.0,    max: 2.0}
        - {name: D_offset_sample,   min: 0.0,     max: 1.0e4}
        - {name: v0,                min: 0.0,     max: 1.0e5}
        - {name: v_beta,            min: -2.0,    max: 2.0}
        - {name: v_offset,          min: -1.0e3,  max: 1.0e3}
        - {name: f0,                min: 0.0,     max: 1.0}
        - {name: f1,                min: -1.0,    max: 1.0}
        - {name: f2,                min: 0.0,     max: 100.0}
        - {name: f3,                min: 0.0,     max: 1.0}
        - {name: phi0_het,          min: -10.0,   max: 10.0}
        - {name: contrast,          min: 0.0,     max: 1.0}
        - {name: offset,            min: 0.5,     max: 1.5}

    optimization:
      nlsq:
        max_nfev: 2000
        ftol: 1.0e-10
        xtol: 1.0e-10
        gtol: 1.0e-10
        cmaes:
          enable: true
        anti_degeneracy:
          enable: true
          per_angle_mode: auto

.. warning::

   ``optimization.nlsq`` is **not** the same as the top-level ``nlsq:`` key used
   by homodyne. Keep the ``optimization:`` parent in heterodyne configs — the
   heterodyne adapter looks there first.

Running the fit
---------------

.. code-block:: python

    from pathlib import Path

    from xpcsjax import ConfigManager, fit_nlsq, load_xpcs_data

    config_path = Path("config_heterodyne.yaml")

    data = load_xpcs_data(str(config_path))
    result = fit_nlsq(data, str(config_path))
    # heterodyne path → a single OptimizationResult (per-angle detail in
    # result.nlsq_diagnostics)

    print(type(result).__name__, float(result.reduced_chi_squared))
    # OptimizationResult 1.07   (illustrative)

Inspecting per-angle results
----------------------------

The joint fit's per-angle quality and scaling live under
``result.nlsq_diagnostics``. The
:func:`~xpcsjax.optimization.nlsq.heterodyne_views.per_angle_chi2` helper reads
the ``chi2_per_angle`` entry as an array, and
:func:`~xpcsjax.optimization.nlsq.heterodyne_views.reconstruct_per_angle_scaling`
recovers the per-angle ``contrast`` / ``offset``.

.. code-block:: python

    import numpy as np

    from xpcsjax.optimization.nlsq.heterodyne_views import per_angle_chi2

    phi = np.asarray(data["phi_angles_list"])
    chi2 = per_angle_chi2(result)  # one entry per angle

    for angle_deg, c in zip(phi, chi2, strict=True):
        print(f"phi={float(angle_deg):7.2f}  chi2_red={float(c): .4e}")

    diag = result.nlsq_diagnostics or {}
    print("physics params:", diag.get("parameter_names"))
    print("contrast/angle:", diag.get("contrast_per_angle"))
    print("offset/angle:  ", diag.get("offset_per_angle"))

The 14 shared physics parameters are the same across all angles and are read
straight off ``result.parameters`` (named in ``diag["parameter_names"]``); only
the per-angle ``contrast`` / ``offset`` scaling varies with angle.

Per-angle reparameterisation in the multi-angle setting
-------------------------------------------------------

With many angles, the per-angle scaling space picks up additional null
directions: angles near ``phi0_het`` are insensitive, angles in the
flow-perpendicular direction are dominant. Layer 1
(:class:`~xpcsjax.optimization.nlsq.per_angle_mode.PerAngleScalingPlan`,
resolved from ``per_angle_mode`` by
:func:`~xpcsjax.optimization.nlsq.per_angle_mode.resolve_per_angle_mode`)
collapses the per-angle scaling onto a small, shared set of parameters
(``constant`` / ``averaged`` / ``individual``), which absorbs that anisotropy
cleanly. Controller activity is recorded on ``result.recovery_actions``.

Next steps
----------

- :doc:`multistart_robust_fit` — combine multi-angle heterodyne with
  multi-start sampling for robustness.
- :doc:`/advanced/cma_es_escape` — CMA-ES is on by default for
  heterodyne fits; this page explains why.
- :doc:`/advanced/anti_degeneracy` — controller layers and their
  cost.
