Installation
============

Requirements
------------

* **Python 3.12+** — declared in ``pyproject.toml``.
* **uv** — strongly recommended for environment management; the project's
  ``Makefile`` assumes ``uv``. See https://docs.astral.sh/uv/.
* A CPU build of JAX. v0.1 sets ``NLSQ_SKIP_GPU_CHECK=1`` and runs CPU-only;
  GPU support is planned for v0.2+.

Quick install (uv, editable)
----------------------------

From a clone of the repository:

.. code-block:: shell

   git clone https://github.com/imewei/xpcsjax.git
   cd xpcsjax
   uv sync                    # creates .venv and installs runtime deps
   uv pip install -e ".[dev]" # editable install with the dev extras

The ``Makefile`` recognises ``uv`` and routes through ``.venv`` automatically.

Documentation extras
--------------------

To build this documentation locally you need the ``docs`` extra, which pulls
in Sphinx, Furo, ``sphinx-copybutton``, ``sphinx-autodoc-typehints``, and
``myst-parser``:

.. code-block:: shell

   uv pip install -e ".[docs]"
   cd docs
   make html
   make htmlview              # opens build/html/index.html in your browser

For a continuous-rebuild workflow:

.. code-block:: shell

   uv pip install sphinx-autobuild
   make -C docs livehtml

Why uv?
-------

The project's ``CLAUDE.md`` mandates uv as the **single source of truth** for
dependency management:

* ``uv.lock`` is the lockfile. Never run a bare ``pip install`` against the
  project venv.
* The ``Makefile`` auto-detects ``uv`` and prefixes the test/lint/typecheck
  commands with ``uv run`` so they route through ``.venv``.

Pip / Poetry / Conda are all *technically* able to install xpcsjax, but they
won't reproduce the locked dependency graph. If you must use pip, install
into a fresh virtualenv from ``pyproject.toml`` and accept that the
characterisation tests are not guaranteed to be bit-equivalent to the
upstream homodyne baselines.

Verifying the install
---------------------

A correct install satisfies four checks:

1. **Import works** without eagerly loading JAX::

      >>> import xpcsjax
      >>> xpcsjax.__version__
      '0.1.0'

   The module exposes ``__all__`` but does not actually pull JAX in until
   you touch one of the lazy attributes — that's the public-API contract
   from :mod:`xpcsjax`.

2. **Float64 is enabled.** ``xpcsjax/__init__.py`` sets
   ``JAX_ENABLE_X64=1`` before the first JAX import. Confirm:

   .. code-block:: python

      import xpcsjax            # triggers env setup
      from xpcsjax import fit_nlsq  # first JAX import lands here
      import jax
      assert jax.config.read("jax_enable_x64") is True

3. **NLSQ is wired.** The fit path depends on ``nlsq>=0.6.10``::

      >>> from nlsq import CurveFit
      >>> CurveFit is not None
      True

4. **Tests pass.** From the repo root:

   .. code-block:: shell

      make test-smoke   # fast subset
      make verify       # lint + advisory mypy + smoke under -x -n auto

   The full characterisation suite (homodyne parity baselines) is env-gated
   behind ``XPCSJAX_RUN_CHARACTERIZATION=1``; see
   :doc:`development/index`.

Optional GPU build (v0.2+)
--------------------------

v0.1 ships CPU-only. To preview GPU paths in a v0.2 development build, set
``NLSQ_SKIP_GPU_CHECK=0`` *before* importing xpcsjax and follow the JAX CUDA
install instructions at https://docs.jax.dev/en/latest/installation.html.
Note that the v0.1 anti-degeneracy controller and CMA-ES escape path are
not yet validated on GPU; expect regressions in the characterisation gate.

Troubleshooting
---------------

* **``ImportError: nlsq``** — install the NLSQ wheel:
  ``uv pip install 'nlsq>=0.6.10'``.

* **``RuntimeError: WorkflowSelector``** — you are calling an NLSQ pre-0.6.0
  symbol; xpcsjax uses ``CurveFit`` directly. Upgrade NLSQ.

* **Slow compile on first call** — the XLA flag
  ``--xla_disable_hlo_passes=constant_folding`` is set automatically in
  ``xpcsjax/__init__.py``. If you've overridden ``XLA_FLAGS`` upstream of
  importing xpcsjax, prepend that flag to your override.

* **``h5py`` import failures** — pin ``h5py>=3.15,<4.0``.
