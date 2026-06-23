"""PySide6 analysis-workbench GUI for xpcsjax.

Import discipline: the GUI process must never import JAX. Modules here import
only Qt + the JAX-free ``xpcsjax.service.events`` schema and (lazily, inside the
worker child) the ``xpcsjax.service`` functions.
"""

from __future__ import annotations
