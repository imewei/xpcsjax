"""JAX-free project model: datasets with an append-only fit-run history.

In-memory for Phase 4; designed to serialize to ``.xpcsproj`` in Phase 6.
Identity is via stable UUID strings (never list position).
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
KILLED = "killed"


@dataclass
class FitRun:
    """One fit attempt for a dataset (append-only history entry)."""

    run_id: str
    status: str
    result_dir: str | None = None
    summary: Any = None
    created_at: str = ""


@dataclass
class Dataset:
    """A config (and its referenced data) plus its run history."""

    dataset_id: str
    config_path: str
    label: str
    runs: list[FitRun] = field(default_factory=list)


@dataclass
class Project:
    """A workbench session: several datasets, each with a run history."""

    datasets: list[Dataset] = field(default_factory=list)

    def add_dataset(self, config_path: str, label: str | None = None) -> Dataset:
        """Append a dataset with a stable id; auto-label from the filename."""
        dataset = Dataset(
            dataset_id=uuid.uuid4().hex,
            config_path=str(config_path),
            label=label or Path(config_path).stem,
        )
        self.datasets.append(dataset)
        return dataset

    def add_run(self, dataset_id: str) -> FitRun:
        """Append a QUEUED run to a dataset; raises KeyError if unknown."""
        dataset = self.dataset_by_id(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        run = FitRun(
            run_id=uuid.uuid4().hex,
            status=QUEUED,
            created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )
        dataset.runs.append(run)
        return run

    def dataset_by_id(self, dataset_id: str) -> Dataset | None:
        """Return the dataset with this id, or None."""
        return next((d for d in self.datasets if d.dataset_id == dataset_id), None)

    def run_by_id(self, run_id: str) -> tuple[Dataset, FitRun] | None:
        """Return ``(dataset, run)`` for this run id, or None."""
        for dataset in self.datasets:
            for run in dataset.runs:
                if run.run_id == run_id:
                    return dataset, run
        return None

    def set_run_status(
        self,
        run_id: str,
        status: str,
        *,
        result_dir: str | None = None,
        summary: Any = None,
    ) -> None:
        """Update a run's status (and optionally its result_dir/summary) in place."""
        found = self.run_by_id(run_id)
        if found is None:
            return
        _, run = found
        run.status = status
        if result_dir is not None:
            run.result_dir = result_dir
        if summary is not None:
            run.summary = summary
