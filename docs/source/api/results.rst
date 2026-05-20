xpcsjax.optimization.nlsq.results
=================================

Dataclasses that describe the output of a fit.

.. currentmodule:: xpcsjax.optimization.nlsq.results

OptimizationResult
------------------

.. autoclass:: xpcsjax.optimization.nlsq.results.OptimizationResult
   :members:
   :no-special-members:



The homodyne path of :func:`xpcsjax.optimization.nlsq.fit_nlsq` returns a single
``OptimizationResult``. Fields:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Field
     - Type
     - Meaning
   * - ``parameters``
     - ``np.ndarray``
     - Fitted parameter vector (in the order declared by ``ConfigManager``).
   * - ``uncertainties``
     - ``np.ndarray``
     - Per-parameter standard deviation from the covariance diagonal.
   * - ``covariance``
     - ``np.ndarray``
     - Full covariance matrix.
   * - ``chi_squared``
     - ``float``
     - Sum of squared residuals.
   * - ``reduced_chi_squared``
     - ``float``
     - ``chi_squared / (n_data - n_params)``.
   * - ``convergence_status``
     - ``str``
     - One of ``'converged'``, ``'max_iter'``, ``'failed'``.
   * - ``iterations``
     - ``int``
     - Number of NLSQ iterations used.
   * - ``execution_time``
     - ``float``
     - Wall-clock fit time in seconds.
   * - ``device_info``
     - ``dict[str, Any]``
     - Captured CPU / device metadata; useful for benchmarking.
   * - ``recovery_actions``
     - ``list[str]``
     - Diagnostic trail of any fallback paths taken (e.g. CMA-ES escape).
   * - ``quality_flag``
     - ``str``
     - One of ``'good'``, ``'warn'``, ``'bad'`` — set by the result builder.
   * - ``streaming_diagnostics``
     - ``dict[str, Any] | None``
     - Per-shard metadata when the HYBRID_STREAMING strategy was used.
   * - ``stratification_diagnostics``
     - ``StratificationDiagnostics | None``
     - Per-angle stratification metadata.
   * - ``nlsq_diagnostics``
     - ``dict[str, Any] | None``
     - NLSQ-level diagnostics (iteration counts, Jacobian rank).
   * - ``sigma_is_default``
     - ``bool``
     - ``True`` when the residual weights fell back to the default σ.

Two convenience properties are exposed:

.. code-block:: python

   result.success    # bool — True if convergence_status == 'converged'
   result.message    # str  — human-readable status summary

Fallback markers
----------------

.. autoclass:: xpcsjax.optimization.nlsq.results.FallbackInfo



.. autoclass:: xpcsjax.optimization.nlsq.results.UseSequentialOptimization



.. autoclass:: xpcsjax.optimization.nlsq.strategies.chunking.StratificationDiagnostics



NLSQResult (heterodyne path)
----------------------------

The heterodyne dispatch returns a ``list[NLSQResult]`` — one element per φ
angle. Its layout mirrors ``OptimizationResult`` for the per-angle subset of
parameters; see :mod:`xpcsjax.optimization.nlsq.heterodyne_results` for the
authoritative dataclass definition.

Heterodyne-specific results
---------------------------

See :mod:`xpcsjax.optimization.nlsq.heterodyne_results` for the
heterodyne-side ``NLSQResult`` dataclass. The autodoc entry is on
:doc:`optimization`.
