Visualization
=============

After fitting with :func:`xpcsjax.fit_nlsq`, generate diagnostic plots and
serialize fitted artifacts with :func:`xpcsjax.generate_nlsq_plots`.

Quick start
-----------

.. code-block:: python

   import xpcsjax

   model  = xpcsjax.HomodyneModel(config_dict)
   data   = xpcsjax.load_xpcs_data(data_path)
   result = xpcsjax.fit_nlsq(model, data, config_dict)

   xpcsjax.generate_nlsq_plots(
       model=model,
       result=result,
       data=data,
       config=config_dict,
       output_dir="results/run_001/",
   )

Output structure
----------------

For each phi angle:

- ``c2_heatmaps_phi_<deg>.png`` — 3-panel comparison
  (Experimental | Fitted | Residuals)
- ``residuals_phi_<deg>.png`` — 4-panel residual diagnostic
  (Residual Map | Distribution + Normal overlay | Diagonal trace |
  Residuals vs Fitted scatter)
- ``simulated_data/simulated_c2_fitted_phi_<deg>deg.png`` — single-panel
  fitted heatmap with annotated stats

Plus, under ``simulated_data/``:

- ``c2_fitted_data.npz`` — compressed numerical arrays. Default LZMA
  compression gives ~30-50% smaller files than DEFLATE on smooth correlation
  data; ``np.load`` reads it transparently. Use ``compression="deflate"`` or
  ``compression="none"`` for faster encoding at the cost of file size.
- ``simulation_config_fitted.json`` — fit parameters, uncertainties,
  reduced chi-squared, convergence status, q value, and per-angle metadata.

NPZ schema
----------

The ``c2_fitted_data.npz`` file contains the following numerical arrays:

================================ ================================== ================
Key                              Shape                              Dtype
================================ ================================== ================
``c2_exp``                       ``(n_phi, n_t1, n_t2)``            float64
``c2_fitted``                    ``(n_phi, n_t1, n_t2)``            float64
``residuals``                    ``(n_phi, n_t1, n_t2)``            float64
``phi_angles``                   ``(n_phi,)``                       float64
``t1``, ``t2``                   ``(n_t1,)``, ``(n_t2,)``           float64
``q``                            scalar                             float64
``params``                       ``(n_params,)``                    float64
``contrast``, ``offset``         scalar                             float64
``reduced_chi_squared``          scalar                             float64
================================ ================================== ================

String metadata (parameter names, analysis mode) lives in the JSON sidecar.

Performance tuning
------------------

For very large datasets, enable ``parallel=True`` to fan out PNG rendering
across ``multiprocessing.cpu_count()`` workers. Model evaluation stays in
the main process (models may not be picklable across spawn boundaries);
only the matplotlib rendering parallelizes.

Datashader-accelerated rendering is opt-in via ``use_datashader=True`` and
requires the ``[viz-fast]`` extra::

   pip install 'xpcsjax[viz-fast]'

Low-level API
-------------

For notebook use or custom layouts, the low-level plot functions return
``matplotlib.figure.Figure`` instances:

- :func:`xpcsjax.viz.plot_nlsq_fit` — 3-panel comparison
- :func:`xpcsjax.viz.plot_residual_map` — 4-panel residual diagnostic
- :func:`xpcsjax.viz.plot_simulated_data` — single-panel fitted heatmap
- :func:`xpcsjax.viz.compute_diagonal_overlay_stats` — diagonal-trace stats

Pass ``save_path=None`` to keep the Figure open for further customization;
pass ``save_path=Path(...)`` to save and close.

Model support
-------------

Currently homodyne-only. ``HeterodyneModel`` raises ``NotImplementedError``
at the orchestrator entry point — heterodyne c2 reconstruction needs
per-angle scaling from ``heterodyne_scaling_utils`` (formulas depend on
analysis mode: constant / auto / fourier / individual), and that
integration is pending.
