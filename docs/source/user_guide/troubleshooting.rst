Troubleshooting
===============

.. currentmodule:: xpcsjax


This page catalogues the failure modes most commonly seen in xpcsjax
analyses and explains the relevant runtime context (environment
variables, lazy import behaviour, anti-degeneracy controller) needed
to interpret each one.

``ImportError`` on ``nlsq`` or ``jax``
--------------------------------------

Symptom
~~~~~~~

.. code-block:: text

   ImportError: cannot import name 'CurveFit' from 'nlsq'
   ImportError: No module named 'jax'

Cause
~~~~~

xpcsjax declares ``nlsq>=0.6.10`` and ``jax`` as runtime dependencies.
If you installed the package outside of the project's ``uv`` lockfile,
or with a stale ``pip install`` that did not resolve the constraints,
these imports can fail.

Fix
~~~

Use the project Makefile target rather than ad-hoc installs:

.. code-block:: console

   $ make dev

which expands to ``uv pip install -e ".[dev]"`` and respects
``uv.lock``. If you are integrating xpcsjax into another project,
ensure ``nlsq>=0.6.10`` is in your environment and that ``jax`` is
installable for your platform.

Slow first compile
------------------

Symptom
~~~~~~~

The first call to :func:`xpcsjax.optimization.nlsq.fit_nlsq` or
:meth:`xpcsjax.core.HomodyneModel.compute_c2` takes much longer than subsequent
calls — sometimes tens of seconds.

Cause
~~~~~

JAX compiles the residual and forward functions via XLA on first
invocation. Compilation is amortised across all subsequent calls with
the same shapes. For ``hybrid-streaming`` strategies on very large
datasets (23M+ correlation entries), the compilation passes are
specifically why xpcsjax disables ``constant_folding`` at package load
time:

.. code-block:: text

   XLA_FLAGS=... --xla_disable_hlo_passes=constant_folding ...

Without this flag, XLA spends > 1 s in constant-folding passes per
JIT cache miss on those large closures.

Fix
~~~

* This is expected behaviour. The second call to the same function
  with the same shapes is fast.
* If first-compile latency is unacceptable in an interactive context,
  consider amortising it by warming the cache with a small synthetic
  call before processing the production data.
* Do **not** disable the XLA flags set in :mod:`xpcsjax` —
  they are tuned for the strategy router's worst-case codepaths.

NaN parameters
--------------

Symptom
~~~~~~~

After a fit, one or more entries in :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.parameters`
is NaN, even though :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.convergence_status` is reported as
``"converged"``.

Cause
~~~~~

This can happen when:

* The initial point landed in a region where the Jacobian has a NaN
  row that was not caught by the anti-degeneracy controller's first
  layer.
* The bounds were so wide that the parameter transform produced an
  overflow in the unconstrained coordinate.
* The input ``c2_exp`` contains NaN entries that were not detected
  during loading.

Fix
~~~

1. Inspect :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.recovery_actions` for entries at
   stage ``"anti_degeneracy"``. If the controller saw the NaN and
   replaced rows, it will say so; if not, the problem entered after
   the controller's hygiene pass.
2. Re-run ``xpcsjax.data.validators.validate_xpcs_data`` against the loaded
   data dictionary to confirm there are no NaNs in ``c2_exp``.
3. Tighten the bounds. Diffusion coefficients spanning more than
   ~6 orders of magnitude in a single fit are unusual and worth
   re-examining.
4. If the failure is reproducible from a clean configuration, file an
   issue with the data shape and the YAML attached.

All-zeros or "flat" fit
-----------------------

Symptom
~~~~~~~

The fit converged, ``quality_flag`` is ``"good"`` or ``"warn"``, but
the fitted parameters are uniformly the initial values (or very close
to them) and the reduced :math:`\chi^2` is suspiciously high.

Cause
~~~~~

The most common cause is bounds that pin the parameters to the
initial values — for example, a lower and upper bound that are
effectively equal, or initial values placed exactly on a bound.

Fix
~~~

* Print bounds and initial values together before the fit:

  .. code-block:: python

     names = cfg.get_active_parameters()
     lo, hi = cfg.get_parameter_bounds()
     init = cfg.get_initial_parameters()
     for n, x0, a, b in zip(names, init, lo, hi):
         print(f"{n:>12s} init={x0:.3e}  bounds=[{a:.3e}, {b:.3e}]")

* Widen the bounds on the offending parameters.
* If the bounds genuinely should be tight, force a multistart with
  ``optimization.nlsq.multistart.n_starts: 8``. If even multistart
  cannot move the parameters, the data is not sensitive to them — a
  scientifically interesting result on its own.

``convergence_status == "max_iter"``
------------------------------------

Symptom
~~~~~~~

The fit ran but did not satisfy the tolerance within the iteration
budget. ``quality_flag`` is typically ``"warn"`` or ``"bad"``.

Cause
~~~~~

* The initial point was far from the solution.
* The bounds are too tight, forcing the trust-region solver to take
  many small steps.
* The model family is mis-specified; the solver is wandering across a
  flat or saddle-like region.

Fix
~~~

1. Raise ``optimization.nlsq.max_iterations``. For very large
   heterodyne fits, 3000–5000 iterations is not unreasonable.
2. Inspect :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.nlsq_diagnostics` for the
   residual history. If it is monotonically decreasing, the run was
   simply truncated and more iterations will likely fix it. If it has
   plateaued, more iterations will not help.
3. Enable multistart. The LHS resampler is the most effective tool
   against initial-condition sensitivity.
4. If multistart fails, the CMA-ES escape will auto-trigger above the
   configured failure threshold; check the ``recovery_actions`` trail
   to confirm it ran.

``convergence_status == "failed"``
----------------------------------

Symptom
~~~~~~~

The trust-region solver reported a hard failure: NaN or Inf in the
Jacobian, an unrecoverable step rejection, or numerical breakdown.

Cause
~~~~~

This is rare and usually indicates a configuration or data error
upstream of the solver. The anti-degeneracy controller typically
catches the issue before the solver throws, but extreme cases — for
example, bounds that contain only non-finite parameter transforms —
fall through.

Fix
~~~

1. Read :attr:`xpcsjax.optimization.nlsq.results.OptimizationResult.recovery_actions` end to end. Every
   intervention is logged with its outcome.
2. Validate the input data:

   .. code-block:: python

      from xpcsjax.data import validate_xpcs_data
      report = validate_xpcs_data(data)
      print(report)

3. Reduce the parameter dimensionality temporarily. Drop to
   ``static_isotropic`` and confirm that fit succeeds; then move up
   to the intended mode.

Lazy import surprises
---------------------

Symptom
~~~~~~~

A static type-checker, or an editor's "Go to definition" feature,
cannot resolve ``xpcsjax.fit_nlsq`` even though the call works at
runtime.

Cause
~~~~~

The public symbols are resolved through a module-level ``__getattr__``
to keep import-time cheap. Some tooling does not honour PEP 562 lazy
attribute resolution.

Fix
~~~

* Import the symbol from its concrete submodule for tooling-friendly
  type resolution:

  .. code-block:: python

     from xpcsjax.optimization.nlsq import fit_nlsq
     from xpcsjax.data import load_xpcs_data
     from xpcsjax.config import ConfigManager

* The literal ``__all__`` in ``xpcsjax/__init__.py`` lists every
  public symbol, so Sphinx autosummary and Pyright's
  ``reportUnsupportedDunderAll`` check both work without
  modification.

JAX environment surprises
-------------------------

Symptom
~~~~~~~

Calculations silently produce ``float32`` results, or GPU-related
warnings appear at import time, or the XLA flag string in your
environment differs from what xpcsjax sets.

Cause
~~~~~

xpcsjax sets three environment variables at package import time, but
only with ``os.environ.setdefault`` (or merging) — never with
unconditional overwrite. If your shell or wrapper script already
exported a conflicting value, xpcsjax does not overwrite it.

The package-level settings are:

* ``JAX_ENABLE_X64=1`` — mandatory; parameters span 6+ orders of
  magnitude.
* ``XLA_FLAGS`` — adds ``--xla_force_host_platform_device_count=4``
  and ``--xla_disable_hlo_passes=constant_folding`` to any existing
  value.
* ``NLSQ_SKIP_GPU_CHECK=1`` — v0.1 is CPU-only.

Fix
~~~

* Confirm the live values inside Python:

  .. code-block:: python

     import os, xpcsjax
     for k in ("JAX_ENABLE_X64", "XLA_FLAGS", "NLSQ_SKIP_GPU_CHECK"):
         print(k, "=", os.environ.get(k))

* If a conflicting value was inherited from the shell, unset it
  before invoking Python:

  .. code-block:: console

     $ unset JAX_ENABLE_X64
     $ python my_fit_script.py

* Never import ``jax`` (directly or indirectly) before importing
  ``xpcsjax``. The environment must be set before the first JAX
  import or the float64 promotion will not take effect.

Enabling verbose logging
------------------------

xpcsjax uses standard ``logging``. To see the strategy router, the
anti-degeneracy controller, and the NLSQ adapter at maximum verbosity,
configure the root logger before the first xpcsjax call:

.. code-block:: python

   import logging
   logging.basicConfig(
       level=logging.DEBUG,
       format="%(asctime)s %(name)s %(levelname)s %(message)s",
   )

   import xpcsjax
   data = xpcsjax.load_xpcs_data("config.yaml")
   result = xpcsjax.fit_nlsq(data, "config.yaml")

For lower-noise diagnostics, set ``INFO`` instead of ``DEBUG``. The
JAX backend loggers are silenced at ``ERROR`` level by xpcsjax's
``__init__.py`` so they do not flood your console with GPU-fallback
warnings on CPU-only systems.

If a fit problem persists after the steps above, capture the result
object (parameters, ``recovery_actions``, all three diagnostic
dictionaries) and the YAML configuration and open an issue.
