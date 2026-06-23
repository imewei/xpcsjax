Installation
============

Requirements
------------

* **Python 3.12+** — declared in ``pyproject.toml``.
* **uv** — strongly recommended for environment management; the project's
  ``Makefile`` assumes ``uv``. See https://docs.astral.sh/uv/.
* A CPU build of JAX. v0.1 sets ``NLSQ_SKIP_GPU_CHECK=1`` and runs CPU-only;
  GPU support is planned for v0.2+.

Quick install (uv)
------------------

The recommended installer is `uv <https://docs.astral.sh/uv/>`_. To get every
feature and avoid missing-dependency issues, install with the ``all`` extra,
which pulls **every** optional dependency — the ``gui``, ``viz-fast``, ``dev``,
``docs``, and ``packaging`` extras combined:

.. code-block:: shell

   uv pip install "xpcsjax[all]"

mamba / conda / other virtual environments
------------------------------------------

Inside an activated ``mamba`` or ``conda`` environment — or any other virtualenv
(``venv``, ``virtualenv``, ``pyenv``) — install with ``pip``:

.. code-block:: shell

   pip install "xpcsjax[all]"

Either installer pulls the core runtime dependencies declared in
``pyproject.toml`` (``jax``, ``nlsq``, ``evosax``, ``h5py``, ``interpax``,
``jaxopt``, ``psutil``, ``scikit-learn``, ``tqdm``, …); the ``all`` extra adds
the GUI (PySide6 + PyQtGraph), fast-viz (datashader), and the full dev/docs/
packaging toolchains (pytest, ruff, mypy, Sphinx, PyInstaller, …) on top.

For a **minimal core install** (NLSQ fitting only, no GUI or fast viz), drop the
extra — ``uv pip install xpcsjax`` / ``pip install xpcsjax``. Individual extras
can also be requested by name, e.g. ``uv pip install "xpcsjax[gui]"`` or
``pip install "xpcsjax[viz-fast]"``.

From source (development)
-------------------------

To contribute or track ``main``, clone the repository and use an editable
install with the ``dev`` extra:

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

GUI extras
----------

The desktop analysis workbench (:doc:`/user_guide/gui`) is optional. Install
the ``gui`` extra for PySide6 + PyQtGraph, and the ``packaging`` extra if you
want to freeze a standalone app with PyInstaller:

.. code-block:: shell

   uv pip install "xpcsjax[gui]"        # PySide6 + PyQtGraph
   pip install "xpcsjax[gui]"           # same, inside conda/mamba/other venvs
   xpcsjax-gui                          # launch the workbench

   uv pip install "xpcsjax[packaging]"  # pyinstaller (frozen-app builds)

The GUI process is JAX-free by design and runs every fit in a separate
``spawn`` worker; see :doc:`/user_guide/gui` for the architecture and the
PyInstaller freeze notes.

Why uv for development?
-----------------------

For **end users**, ``pip install xpcsjax`` into a conda/mamba/venv environment
is fully supported — the section above is all you need. The note here only
concerns *contributing* to xpcsjax from a source clone.

The project's ``CLAUDE.md`` mandates uv as the **single source of truth** for
the development dependency graph:

* ``uv.lock`` is the lockfile. Never run a bare ``pip install`` against the
  project venv.
* The ``Makefile`` auto-detects ``uv`` and prefixes the test/lint/typecheck
  commands with ``uv run`` so they route through ``.venv``.

Pip / Poetry / Conda are all *technically* able to install the dev environment,
but they won't reproduce the locked dependency graph. If you must use pip for a
dev checkout, install into a fresh virtualenv from ``pyproject.toml`` and accept
that the synthetic parity tests are not guaranteed to reproduce xpcsjax's pinned
engine output bit-for-bit on an unlocked dependency graph.

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

   The synthetic parity tests live under :file:`tests/parity/` and run
   as part of the normal suite; see :doc:`development/index`.

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
