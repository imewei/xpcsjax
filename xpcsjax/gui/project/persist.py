"""JSON save/load for a workbench session (.xpcsproj). JAX-free, stdlib only."""

from __future__ import annotations

import json
from pathlib import Path

from xpcsjax.gui.project.model import Dataset, FitRun, Project

_SCHEMA = "xpcsjax.project/v1"


def _rel(path: str, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base))
    except ValueError:
        return str(path)  # different drive / outside the tree -> keep absolute


def _abs(stored: str, base: Path) -> str:
    p = Path(stored)
    return str(p if p.is_absolute() else (base / p))


def save_project(project: Project, path: str | Path) -> None:
    """Serialize ``project`` to ``path`` as JSON (paths stored relative to the file).

    Invariant: dataset/run paths are **absolute** (the GUI stores absolute paths
    from file dialogs, and Plan-F ``add_dataset`` stores ``str(config_path)``).
    They are written relative to the project file for portability and resolved
    back against the project-file directory on load.
    """
    path = Path(path)
    base = path.resolve().parent
    payload = {
        "schema": _SCHEMA,
        "datasets": [
            {
                "dataset_id": d.dataset_id,
                "config_path": _rel(d.config_path, base),
                "label": d.label,
                "runs": [
                    {
                        "run_id": r.run_id,
                        "status": r.status,
                        "result_dir": _rel(r.result_dir, base) if r.result_dir else None,
                        "created_at": r.created_at,
                    }
                    for r in d.runs
                ],
            }
            for d in project.datasets
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> Project:
    """Reconstruct a :class:`Project` from a ``.xpcsproj`` file.

    Run summaries are left ``None`` (re-loaded lazily from ``result_dir`` via the
    Plan-D ``load_result_summary``). Raises ``ValueError`` on a bad schema.
    """
    path = Path(path)
    base = path.resolve().parent
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))  # JSONDecodeError IS a ValueError
    except OSError as exc:
        raise ValueError(f"could not read .xpcsproj: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError(f"unrecognized .xpcsproj schema: {payload.get('schema')!r}")

    try:
        project = Project()
        for ds in payload["datasets"]:
            dataset = Dataset(
                dataset_id=str(ds["dataset_id"]),
                config_path=_abs(str(ds["config_path"]), base),
                label=str(ds.get("label", "")),
            )
            for r in ds.get("runs", []):
                dataset.runs.append(
                    FitRun(
                        run_id=str(r["run_id"]),
                        status=str(r["status"]),
                        result_dir=_abs(str(r["result_dir"]), base) if r.get("result_dir") else None,
                        created_at=str(r.get("created_at", "")),
                    )
                )
            project.datasets.append(dataset)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed .xpcsproj structure: {exc}") from exc
    return project
