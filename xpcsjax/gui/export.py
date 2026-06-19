"""JAX-free figure-export helper for the GUI.

Copies the publication figures the fit worker already wrote under
``<result_dir>/plots`` into a user-chosen destination directory.

Design constraints
------------------
- **No JAX** — pure stdlib + shutil; safe to call from the GUI process.
- **No re-render** — the worker has already run ``generate_nlsq_plots`` and
  written PNG/PDF artifacts; this module only *copies* them.
- **No worker** — synchronous file copy is instantaneous on local disk.
- Missing or empty ``plots/`` directory returns ``[]`` without raising.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def export_figures(
    result_dir: str | Path,
    dest_dir: str | Path,
) -> list[Path]:
    """Copy publication figures from ``<result_dir>/plots`` to ``dest_dir``.

    Copies every ``*.png`` and ``*.pdf`` found under the ``plots/``
    sub-directory of *result_dir* (recursively, so ``simulated_data/`` and
    any other sub-folder are included) into *dest_dir*.

    Parameters
    ----------
    result_dir:
        The run's result directory (contains a ``plots/`` sub-directory
        written by the fit worker).
    dest_dir:
        Destination directory.  Created if it does not exist.

    Returns
    -------
    list[Path]
        Absolute paths to the *copied* files inside *dest_dir*.
        Empty list when ``plots/`` is absent, empty, or contains no
        ``*.png``/``*.pdf`` files.
    """
    plots_dir = Path(result_dir) / "plots"
    if not plots_dir.is_dir():
        return []

    # Collect all .png / .pdf recursively
    sources = [
        p for p in plots_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".pdf"}
    ]
    if not sources:
        return []

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    used_names: set[str] = set()
    for src in sources:
        candidate = src.name
        if candidate in used_names:
            # Disambiguate with the immediate parent directory name.
            candidate = f"{src.parent.name}__{src.name}"
        # If still colliding (two sub-dirs share both name and parent-name),
        # fall back to a numeric suffix until unique.
        base_candidate = candidate
        counter = 1
        while candidate in used_names:
            stem = Path(base_candidate).stem
            suffix = Path(base_candidate).suffix
            candidate = f"{stem}__{counter}{suffix}"
            counter += 1
        used_names.add(candidate)
        dst = dest / candidate
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied
