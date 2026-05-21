Visualization (``xpcsjax.viz``)
================================

The ``xpcsjax.viz`` subpackage provides diagnostic plots and artifact
serialization for NLSQ fit results. All public symbols are lazy-loaded to
keep matplotlib off the import path until first use.

.. currentmodule:: xpcsjax.viz

Public surface
--------------

The orchestrator and the three low-level plot functions are the primary
user-facing interface. All four accept both
:class:`~xpcsjax.core.HomodyneModel` and
:class:`~xpcsjax.core.HeterodyneModel` (the heterodyne path requires
``individual`` per-angle scaling layout — see :ref:`viz-heterodyne`).

Orchestrator
~~~~~~~~~~~~

.. autofunction:: xpcsjax.viz.nlsq_plots.generate_nlsq_plots

Low-level plot functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: xpcsjax.viz.nlsq_plots.plot_nlsq_fit

.. autofunction:: xpcsjax.viz.nlsq_plots.plot_residual_map

.. autofunction:: xpcsjax.viz.nlsq_plots.plot_simulated_data

Diagnostics
~~~~~~~~~~~

.. autoclass:: xpcsjax.viz.diagnostics.DiagonalOverlayResult
   :members:
   :undoc-members:

.. autofunction:: xpcsjax.viz.diagnostics.compute_diagonal_overlay_stats

.. _viz-heterodyne:

Heterodyne support
------------------

Both model types are accepted by all viz functions. For
:class:`~xpcsjax.core.HeterodyneModel`, the ``individual`` per-angle scaling
layout is required:

.. code-block:: text

   result.parameters = [c_0 … c_{n-1}, o_0 … o_{n-1}, physical_0 … physical_13]

This layout is produced when fitting with ``per_angle_mode="individual"``.
Results from ``constant``, ``fourier``, or ``auto`` modes (when auto resolves
to non-individual) will raise :class:`NotImplementedError` at render time with
a message identifying the parameter-count mismatch.

To plot a heterodyne fit from a non-individual mode, refit with
``per_angle_mode="individual"`` or use the upstream ``heterodyne`` package.
Full mode parity is scheduled for v0.2.

.. _viz-datashader:

Datashader backend
------------------

Install the optional fast backend with::

   pip install 'xpcsjax[viz-fast]'

When installed and ``use_datashader=True`` (the default), the 3-panel
comparison plot is rendered via a hybrid pipeline: Datashader rasterizes the
raw c₂ arrays to a 1200 px image, then matplotlib adds axes, colorbars, and
titles. This is 5–10× faster per call and avoids matplotlib memory pressure on
large grids (≳10⁶ samples per angle).

The ``"residuals"`` and ``"simulated"`` plot families always render through
matplotlib regardless of backend, because those layouts (histogram, scatter,
single-panel heatmap) do not benefit from Datashader rasterization.

When the extra is missing, :func:`~xpcsjax.viz.nlsq_plots.generate_nlsq_plots`
logs a warning and silently falls back to matplotlib — no manual intervention
required.

.. autoclass:: xpcsjax.viz.datashader_backend.DatashaderRenderer
   :members: rasterize_heatmap

.. autofunction:: xpcsjax.viz.datashader_backend.plot_c2_heatmap_fast

.. autofunction:: xpcsjax.viz.datashader_backend.plot_c2_comparison_fast

.. _viz-artifacts:

Artifact schema
---------------

:func:`~xpcsjax.viz.nlsq_plots.generate_nlsq_plots` writes two files per
fit under ``output_dir/simulated_data/``:

``c2_fitted_data.npz``
   Compressed NumPy archive (LZMA by default, DEFLATE-9 on LZMA failure).
   Keys: ``c2_exp``, ``c2_fitted``, ``residuals``, ``phi_angles``, ``t1``,
   ``t2``, ``q``, ``params``, ``contrast``, ``offset``,
   ``reduced_chi_squared``.

``simulation_config_fitted.json``
   Human-readable metadata. Top-level keys:

   * ``fit`` — ``parameters`` (values / uncertainties / names),
     ``contrast``, ``offset``, ``reduced_chi_squared``,
     ``convergence_status``, ``iterations``, ``execution_time``.
   * ``physics`` — ``q_value_angstrom_inv``, ``stator_rotor_gap_angstrom``,
     ``dt``, ``analysis_mode``.
   * ``data`` — ``n_phi``, ``n_t1``, ``n_t2``, ``phi_angles_deg``.

Both files use atomic rename (write to a ``.tmp`` sibling, then
:func:`os.replace`) for crash safety.
