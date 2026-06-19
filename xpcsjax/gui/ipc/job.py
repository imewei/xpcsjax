"""The picklable fit-job spec passed across the spawn boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FitJob:
    """A self-contained fit request. All fields are picklable primitives.

    ``overrides`` is a plain dict (mapping to ``service.fit.FitOverrides``
    fields) so this module stays free of any JAX-bearing import; the worker
    converts it to ``FitOverrides`` in the child. Because the whole job crosses
    the spawn boundary, every value (including each ``overrides`` value) must be
    a picklable primitive — plain numbers / bools / strings, no live objects.
    """

    run_id: str
    config_path: str
    output_dir: str | None = None
    output_format: str = "both"
    overrides: dict[str, Any] | None = None
    phi_subset: tuple[float, ...] | None = None
    make_plots: bool = True
