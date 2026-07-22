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

The main window is a logic-free view driven by the ``FitQueueController``.
The workflow is **config-first and toolbar-driven** — there is no tabbed
config/data/fit setup area; the central area shows only the per-angle fitting
results. A quick-access toolbar carries every operational action, in flow
order, each with a keyboard shortcut and hover tooltip:

**Create Config** (``Ctrl+Shift+N``)
   Generate a new YAML config from one of the four mode templates via a
   dialog (numeric fields show placeholder format examples); an
   overwrite-confirmation prompt guards an existing file at the same path.

**Edit Config** (``Ctrl+E``)
   Open the selected config in a raw-YAML text editor that validates syntax
   before writing, instead of silently saving invalid YAML.

**Load Config** (``Ctrl+L``)
   Add a config as a new *dataset* to the project (see below); the freshly
   loaded dataset is auto-selected as Run's target unless a run is already
   active or being viewed, in which case the target is left alone.

**Run** (``F5``) / **Cancel** (``Shift+F5``)
   Enqueue a fit for the active dataset, or cancel the selected run. Run and
   Cancel sit in their own toolbar groups (not flush against each other) and
   Cancel always asks for confirmation first — cancelling discards a
   possibly multi-minute fit with no undo.

**Export Figure** (``Ctrl+Shift+E``)
   Copy the selected run's ``*.png`` / ``*.pdf`` plot files to a chosen
   directory.

A persistent status-bar label always names which dataset Run currently
targets, and the status pill (idle / running / finished / failed) plus a
streaming **Fitting Process** log dock report progress. Docks:

**Project sidebar**
   A *datasets → runs* tree (multi-select). Selecting a run drives the
   central panel and the Inspector; selecting a dataset row retargets Run.

**Comparison** / **Inspector** (tabbed together)
   Comparison renders a real side-by-side table for up to two selected runs,
   marking disagreeing rows with ``≠``. Inspector shows the fitted
   parameters / uncertainties table, the diagnostics tree (top level
   expanded only, by default), and the fit summary (reduced
   :math:`\chi^2`, iteration count, strategy) for the run currently shown.

**Central results area**
   A stacked view: the per-phi results grid (Exp/Fitted/Residual maps with a
   shared color-bar legend per view, plus a "Jump to φ" navigator above 8
   angles) when a viz bundle is available, otherwise a plain-text summary.
   Built from the fit's own JAX-free artifact bundle, so viewing a result
   never imports JAX into the GUI process. See
   :doc:`/user_guide/interpreting_results` and :doc:`/advanced/anti_degeneracy`
   for what the diagnostics mean.

**Inspect Data File…** (File menu, ``Ctrl+I``)
   A read-only HDF5 metadata browser (h5py only — it does *not* import the
   JAX-bearing loader) plus a two-time :math:`C_2` preview, independent of
   any loaded project dataset. Large arrays are block-mean rasterised for
   display only.

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
