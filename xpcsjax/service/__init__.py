"""Headless core-service layer for xpcsjax.

This package is the argparse-free, Qt-free orchestration seam shared by the CLI
and (in later phases) the GUI worker. **Import discipline:** this ``__init__``
must stay free of eager imports that pull in ``jax`` / ``xpcsjax.core`` so that
JAX-free consumers (e.g. ``xpcsjax.service.events``) can be imported without
loading JAX. Heavier submodules (``fit``, ``data``, ``plots``) are added in
Plan B and must be imported directly by callers, not re-exported here.

Import boundary (audited 2026-06-18)
------------------------------------
JAX-free in-process (GUI-importable): ``xpcsjax.service.events`` and
``xpcsjax.config`` (after the parameter_manager lazy-import fix). NOT JAX-free,
by design: ``xpcsjax.data`` (the loader emits JAX arrays via
``xpcsjax.data.xpcs_loader``). Data loading therefore runs worker-side in later
phases; a JAX-free HDF5 *metadata-only* reader for the GUI preview is deferred
to Phase 2, not Phase 1A.
"""

from __future__ import annotations
