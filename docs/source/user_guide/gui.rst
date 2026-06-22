Analysis workbench (GUI)
========================

xpcsjax ships a desktop **analysis workbench** — a PySide6 graphical front-end
for loading data, editing a configuration, running an NLSQ fit, and inspecting
the result without writing a script. It wraps the same fit path as the
command-line :doc:`/user_guide/cli`; nothing in the GUI changes the numerics.

Launching
---------

The workbench is registered as a console script (with the usual ``xj`` alias):

.. code-block:: console

   $ xpcsjax-gui            # full name
   $ xj-gui                 # short alias

It recognises ``--help`` / ``--version`` like the sibling commands; any other
argument is forwarded to Qt, so a headless smoke run is possible:

.. code-block:: console

   $ xpcsjax-gui -platform offscreen

The GUI is an **optional extra** — install it with the ``gui`` extra (PySide6 +
PyQtGraph), see :doc:`/installation`:

.. code-block:: shell

   uv pip install "xpcsjax[gui]"   # or: pip install "xpcsjax[gui]"

Architecture: the GUI never imports JAX
---------------------------------------

The single most important property of the workbench is an **import-discipline
invariant**: the GUI process imports only Qt and the JAX-free
:mod:`xpcsjax.service.events` schema. It **never** imports JAX. Every fit runs
in a separate, ``spawn``-launched worker process that lazily imports JAX and
the :doc:`service layer </api/service>` *inside the child*.

.. code-block:: text

   GUI process (Qt only, no JAX)
     │  FitQueueController  ──spawn──▶  worker process (imports JAX + xpcsjax.service)
     │                                    │  service.fit.run_fit(...)
     │  ◀── FitEvent stream (Started / Iteration / LayerStatus / Banner /
     │         LogLine / Finished / Failed / Died) ──────────────────────┘
     ▼
   Views update from controller signals

This keeps the UI responsive (the multi-second JAX warm-up and the CPU-bound
solve happen off the GUI thread), isolates a crashing or out-of-memory fit from
the window, and lets the bounded-concurrency queue run several fits at once.
The worker streams structured progress events back to the parent, which bridges
them onto Qt signals; see :class:`xpcsjax.service.events.FitEvent` and its
subclasses for the wire schema.

Workbench layout
----------------

The main window is a logic-free view driven by the ``FitQueueController``. It
is organised into tabs plus docks:

**Config tab**
   A configuration editor with a form view and a raw-YAML toggle, backed by
   **live JAX-free validation** (:func:`xpcsjax.service.config.validate_config`).
   Invalid fields are flagged as you type, before any fit is launched.

**Data tab**
   A JAX-free HDF5 metadata browser (h5py only — it does *not* import the
   JAX-bearing loader) plus a two-time :math:`C_2` preview of the selected
   dataset. Large arrays are block-mean rasterised for display only.

**Fit tab**
   A read-only summary of the resolved configuration and any overrides that
   will be sent to the worker, and the controls to enqueue the fit.

**Inspector dock**
   The fitted parameters / uncertainties table, the diagnostics tree, and the
   fit summary (reduced :math:`\chi^2`, iteration count, strategy).

**Live diagnostics**
   A streaming view of the in-progress fit: the SSR curve, the L1–L5
   anti-degeneracy layer status chips, and the banner log — fed by the
   worker's event stream. See :doc:`/user_guide/interpreting_results` and
   :doc:`/advanced/anti_degeneracy` for what the layers mean.

**Interactive plots**
   PyQtGraph two-time, residual, and diagonal-overlay views, built from the
   fit's own JAX-free artifact bundle, so plotting never re-imports JAX into
   the GUI process.

**Project sidebar**
   A *datasets → runs* tree plus a side-by-side comparison view for inspecting
   two runs together.

Projects (``.xpcsproj``)
------------------------

The workbench keeps an in-memory **project**: a set of datasets, each with an
append-only history of fit runs. A project is saved through the **File menu**
as a ``.xpcsproj`` file — plain JSON, written with an atomic replace, holding
no JAX state. Re-opening a project restores the session (datasets, config, and
run history). Each fit run writes to its own per-run output directory so reruns
never clobber earlier results.

Exporting
---------

Figures are exported through a JAX-free helper, so saving a plot from the GUI
does not pay the JAX import cost. Numerical results are written by the same
:doc:`service persistence layer </api/service>` the CLI uses
(:func:`xpcsjax.service.persist.save_results`), in JSON, NPZ, or both.

Packaging a standalone app
--------------------------

The workbench can be frozen into a self-contained desktop app with
PyInstaller. Install the ``packaging`` extra and build the bundled spec:

.. code-block:: shell

   uv pip install -e ".[packaging]"
   uv run pyinstaller packaging/xpcsjax-gui.spec --noconfirm

A few constraints are baked into the entry point and the spec:

* The build is **per-OS** — build on the platform you ship to.
* A **one-dir** bundle is used, not one-file: one-file re-extracts the whole
  app on every process launch, including each ``multiprocessing`` spawn worker,
  which is slow and can race with the JAX/datashader data footprint.
* :func:`multiprocessing.freeze_support` is called *first* in
  :func:`xpcsjax.gui.app.main` so spawn workers in a frozen app do not re-run
  the entry point.

See ``packaging/README.md`` in the repository for the full freeze notes.

See also
--------

* :doc:`/user_guide/cli` — the command-line equivalent of every fit the GUI
  runs.
* :doc:`/api/service` — the JAX-free / worker-side service layer the GUI and
  CLI share.
* :doc:`/user_guide/interpreting_results` — reading the fitted parameters and
  diagnostics the Inspector shows.
