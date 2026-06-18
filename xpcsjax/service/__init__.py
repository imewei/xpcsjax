"""Headless core-service layer for xpcsjax.

This package is the argparse-free, Qt-free orchestration seam shared by the CLI
and (in later phases) the GUI worker. **Import discipline:** this ``__init__``
must stay free of eager imports that pull in ``jax`` / ``xpcsjax.core`` so that
JAX-free consumers (e.g. ``xpcsjax.service.events``) can be imported without
loading JAX. Heavier submodules (``fit``, ``data``, ``plots``) are added in
Plan B and must be imported directly by callers, not re-exported here.
"""

from __future__ import annotations
