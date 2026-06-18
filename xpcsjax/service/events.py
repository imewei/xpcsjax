"""Structured fit-progress events streamed from a worker to the GUI.

This module is deliberately dependency-light: it imports only the standard
library so it can be imported by the JAX-free GUI process *and* pickled across
a ``multiprocessing`` (spawn) boundary. It must never import ``jax``,
``xpcsjax.core``, Qt, or ``h5py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BannerKind(StrEnum):
    """Category of a diagnostic banner emitted during a fit."""

    CMAES_ESCAPE = "cmaes_escape"
    GRADIENT_COLLAPSE = "gradient_collapse"
    INFO = "info"


@dataclass(frozen=True)
class FitEvent:
    """Base class for all fit events.

    Every event carries the originating ``run_id`` and a monotonic per-run
    sequence number ``seq`` so a shared reader can demultiplex multiple
    concurrent runs and order events within a single run.
    """

    # NOTE: keep every field non-default. Frozen-dataclass inheritance + pickle
    # rely on it — giving run_id/seq a default would force every subclass field
    # to have a default too (dataclass field-ordering / MRO rule).
    run_id: str
    seq: int


@dataclass(frozen=True)
class Started(FitEvent):
    """A fit has begun."""

    mode: str
    settings_summary: str


@dataclass(frozen=True)
class Iteration(FitEvent):
    """One solver iteration's convergence sample."""

    n: int
    ssr: float
    chi2: float


@dataclass(frozen=True)
class LayerStatus(FitEvent):
    """Anti-degeneracy layer activation snapshot (L1-L5)."""

    # `layers` is a dict: the dataclass is only top-level frozen, contents stay
    # mutable. For Phase 1A the IPC contract is pickle round-trip + equality only
    # (both hold); deep immutability (tuple-of-pairs / MappingProxyType) is
    # deferred and NOT required.
    layers: dict[str, bool]
    mode: str


@dataclass(frozen=True)
class Banner(FitEvent):
    """A diagnostic banner (e.g. CMA-ES escape, gradient collapse)."""

    text: str
    kind: BannerKind


@dataclass(frozen=True)
class LogLine(FitEvent):
    """A forwarded log record (level + message strings only)."""

    level: str
    msg: str


@dataclass(frozen=True)
class Finished(FitEvent):
    """Terminal: the fit completed; result written to ``result_path``."""

    result_path: str


@dataclass(frozen=True)
class Failed(FitEvent):
    """Terminal: the fit raised; ``traceback`` is the formatted exception."""

    traceback: str


@dataclass(frozen=True)
class Died(FitEvent):
    """Terminal (synthetic, parent-emitted): the worker exited abnormally."""

    exit_code: int | None
    signal: int | None


#: Event types that end a run. A bounded queue must never drop these.
TERMINAL_EVENTS: tuple[type[FitEvent], ...] = (Finished, Failed, Died)
