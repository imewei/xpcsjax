Laminar Flow Fit (7 Parameters)
===============================

.. currentmodule:: xpcsjax


The ``laminar_flow`` analysis mode extends the static homodyne model
with a shear term, producing seven free parameters: three diffusion
coefficients (``D0``, ``alpha``, ``D_offset``) and four shear
coefficients (``gamma_dot_t0``, ``beta``, ``gamma_dot_t_offset``,
``phi0``). The added shear modes are weakly identified from the
correlation function alone, which is why the
:doc:`anti-degeneracy controller </advanced/anti_degeneracy>` is
critical in this mode.

Configuration
-------------

A representative seven-parameter config:

.. code-block:: yaml

    # config_laminar_flow.yaml
    analysis_settings:
      analysis_mode: laminar_flow

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
      values: [1.0e3, -1.5, 1.0e2, 1.0e-3, 0.0, 0.0, 0.0]
      parameter_names:
        [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

    parameter_space:
      bounds:
        - {name: D0,                  min: 1.0,    max: 1.0e6}
        - {name: alpha,               min: -2.0,   max: 2.0}
        - {name: D_offset,            min: 0.0,    max: 1.0e4}
        - {name: gamma_dot_t0,        min: 0.0,    max: 1.0}
        - {name: beta,                min: -2.0,   max: 2.0}
        - {name: gamma_dot_t_offset,  min: 0.0,    max: 1.0}
        - {name: phi0,                min: -90.0,  max: 90.0}

    optimization_config:
      angle_filtering:
        enabled: true
        target_ranges:
          - {min_angle: -10.0, max_angle: 10.0}
          - {min_angle: 170.0, max_angle: 190.0}
        fallback_to_all_angles: true

    anti_degeneracy:
      enabled: true
      fourier_reparam:
        enabled: true
      shear_weighting:
        enabled: true

.. note::

   ``angle_filtering`` keeps only the angles where the shear modulation
   is large enough to constrain the four shear parameters. With
   ``fallback_to_all_angles: true`` xpcsjax will still attempt a fit on
   the full angular range if no target angles are matched.

Running the fit
---------------

The call pattern is identical to the static case — the engine inspects
the YAML to decide which parameters are active.

.. code-block:: python

    from pathlib import Path

    from xpcsjax import ConfigManager, fit_nlsq, load_xpcs_data

    config_path = Path("config_laminar_flow.yaml")

    data = load_xpcs_data(str(config_path))
    cm = ConfigManager(str(config_path))
    cm.load_config()

    print(cm.get_active_parameters())
    # ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta',
    #  'gamma_dot_t_offset', 'phi0']

    result = fit_nlsq(data, cm)

Reading the shear parameters
----------------------------

Because the parameter order is fixed by the YAML, slicing
``result.parameters`` against ``get_active_parameters()`` is the
clearest way to report results:

.. code-block:: python

    names = cm.get_active_parameters()
    for name, value, sigma in zip(
        names, result.parameters, result.uncertainties, strict=True
    ):
        print(f"{name:>22}: {float(value): .6e} ± {float(sigma): .2e}")

A representative output (illustrative, not from any real dataset):

.. code-block:: text

                       D0:  1.124e+03 ± 8.41e+00
                    alpha: -1.487e+00 ± 4.10e-03
                 D_offset:  9.812e+01 ± 1.20e+00
             gamma_dot_t0:  4.230e-03 ± 6.10e-05
                     beta:  1.020e+00 ± 3.30e-02
       gamma_dot_t_offset:  0.000e+00 ± 1.80e-04
                     phi0:  1.234e+01 ± 9.50e-02

The relatively large uncertainty on ``gamma_dot_t_offset`` is typical:
in many datasets it is degenerate with ``gamma_dot_t0`` and the
controller has down-weighted it.

What the anti-degeneracy controller is doing
--------------------------------------------

In ``laminar_flow`` mode the engine activates the full five-layer
controller documented at :doc:`/advanced/anti_degeneracy`. The two
layers that matter most here are:

- :class:`~xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer`
  rewrites the shear parameter sub-space in a basis where directions
  weakly constrained by data have small singular values. The trust
  region solve then handles them naturally without inflating the
  condition number.
- :class:`~xpcsjax.optimization.nlsq.shear_weighting.ShearSensitivityWeighting`
  applies a per-angle weight derived from ``cos(2(phi - phi0))`` so
  that low-sensitivity angles contribute less to the Jacobian.

Audit trail
~~~~~~~~~~~

The actions taken by the controller in this run are listed on
``result.recovery_actions``:

.. code-block:: python

    for action in result.recovery_actions:
        print(action)

If CMA-ES was triggered (see :doc:`/advanced/cma_es_escape`), it will
appear as an entry here.

Next steps
----------

- :doc:`heterodyne_multiangle` — switch to the two-component
  heterodyne model.
- :doc:`/advanced/anti_degeneracy` — detailed description of each
  layer of the controller.
- :doc:`/theory/anti_degeneracy` — derivation and motivation.
