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

Two backends are wired into :func:`xpcsjax.generate_nlsq_plots`:

* **Datashader (default fast path)** — used when ``use_datashader=True``
  (default) and the ``[viz-fast]`` extra is installed. Renders the 3-panel
  comparison plot via a hybrid pipeline: Datashader rasterizes the raw
  c2 arrays to an 800-1200 px image on the CPU, then matplotlib displays
  the pre-rasterized image and adds colorbars / axes / titles. This keeps
  the matplotlib path tiny no matter how large the raw grid is — making
  it the only viable backend for ≳10⁶ samples per angle (e.g. 1000×1000
  t₁×t₂ surfaces) where bare matplotlib ``imshow`` becomes prohibitive.

  Per-call speedup: 5-10× over matplotlib. Combined with parallel
  multiprocessing across angles, cumulative speedup on a many-core box is
  ~50-200× over sequential matplotlib.

* **matplotlib (publication-quality fallback)** — used when
  ``use_datashader=False`` or the ``[viz-fast]`` extra is missing. Produces
  the full plot family (3-panel comparison, 4-panel residual diagnostic,
  single-panel simulated heatmap) at full matplotlib fidelity.

The ``parallel=True`` flag (default) dispatches the per-angle render across
a ``multiprocessing.Pool`` using the ``spawn`` start method. The pool size
is ``min(cpu_count(), n_phi)``. Model evaluation stays in the main process
(models may not be picklable across spawn boundaries); only the rendering
parallelizes.

The Datashader backend is the primary motivation for parallelism — a
single Datashader render is already fast, but with 50+ angles you want to
fan out. The matplotlib path also honours ``parallel=True``, but the
absolute speedup is smaller because matplotlib's per-call cost is already
low and IPC overhead eats most of the gain.

Install the fast extras with::

   pip install 'xpcsjax[viz-fast]'

When the extra is missing and ``use_datashader=True`` (the default), the
orchestrator logs a warning and silently falls back to matplotlib — no
manual intervention required. Plot families "residuals" and "simulated"
always render through matplotlib regardless of backend choice, since
those layouts (histogram, scatter, single-panel) don't benefit from
Datashader's rasterization.

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

Both :class:`xpcsjax.HomodyneModel` and :class:`xpcsjax.HeterodyneModel`
are supported, with one caveat: heterodyne plotting in v0.1 requires the
``individual`` per-angle scaling layout

.. code-block:: text

   result.parameters = [c_0..n_phi-1, o_0..n_phi-1, physical_0..13]

The orchestrator validates this upfront before any rendering starts.
Heterodyne results from the ``constant``, ``fourier``, or ``auto`` scaling
modes will raise :class:`NotImplementedError` with a clear message naming
the parameter-count mismatch. Full mode parity (``constant`` /
``fourier``) is scheduled for v0.2; in the meantime, refit with
``per_angle_mode="individual"`` if you need plotting, or pin the upstream
``heterodyne`` package for non-individual workflows.

The 4-layer anti-degeneracy contract for heterodyne fitting is unaffected
by this restriction — see :doc:`../theory/heterodyne_anti_degeneracy`.
