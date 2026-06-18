"""Headless NLSQ fit service — the argparse-free core of the CLI fit path.

Worker-side module: it imports JAX (via :func:`xpcsjax.fit_nlsq`) and must only
be imported by the fit worker, never by the GUI process. Do NOT re-export it
from ``xpcsjax.service.__init__`` (that would defeat the JAX-free import guard
on ``xpcsjax.service.events``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.config.manager import ConfigManager

logger = get_logger(__name__)

_NLSQ_SECTION = ("optimization", "nlsq")


def _set_nested(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set ``cfg[path[0]][path[1]]...`` = value, creating dicts as needed.

    Parameters
    ----------
    cfg : dict[str, Any]
        The top-level config dict to mutate.
    path : tuple[str, ...]
        Sequence of keys describing the nested path.
    value : Any
        The value to store at the leaf key.
    """
    node = cfg
    for key in path[:-1]:
        existing = node.get(key)
        if not isinstance(existing, dict):
            existing = {}
            node[key] = existing
        node = existing
    node[path[-1]] = value


@dataclass(frozen=True)
class FitOverrides:
    """Typed, argparse-free form of the CLI's NLSQ runtime knobs.

    Attributes
    ----------
    multistart : bool or None
        Enable or disable multi-start NLSQ. ``None`` leaves the config
        key untouched.
    multistart_n : int or None
        Number of random starts. ``None`` leaves the config key untouched.
    max_iterations : int or None
        Maximum solver iterations. ``None`` leaves the config key untouched.
    tolerance : float or None
        Convergence tolerance applied to ``ftol``, ``xtol``, and ``gtol``
        simultaneously. ``None`` leaves all three config keys untouched.
    verbose : bool
        When ``True`` (and ``quiet`` is ``False``), sets ``nlsq.verbose=2``.
    quiet : bool
        When ``True``, sets ``nlsq.verbose=0`` (wins over ``verbose``).
    """

    multistart: bool | None = None
    multistart_n: int | None = None
    max_iterations: int | None = None
    tolerance: float | None = None
    verbose: bool = False
    quiet: bool = False


def apply_overrides(config_manager: ConfigManager, overrides: FitOverrides) -> None:
    """Merge ``overrides`` into ``config_manager.config`` in place.

    Behaviour-identical to the legacy
    ``cli.optimization_runner.apply_cli_overrides``; only the input shape
    changed (typed dataclass instead of ``argparse.Namespace``).

    Parameters
    ----------
    config_manager : ConfigManager
        The config manager whose ``.config`` dict is mutated. If
        ``.config`` is not a :class:`dict`, this function is a no-op.
    overrides : FitOverrides
        Typed overrides to apply. Fields that are ``None`` (or ``False``
        for the bool verbosity flags) are silently skipped.
    """
    cfg = config_manager.config
    if not isinstance(cfg, dict):
        return

    if overrides.multistart is not None:
        _set_nested(cfg, (*_NLSQ_SECTION, "multi_start", "enable"), bool(overrides.multistart))
        logger.info("Override: multi_start.enable = %s", bool(overrides.multistart))

    if overrides.multistart_n is not None:
        _set_nested(cfg, (*_NLSQ_SECTION, "multi_start", "n_starts"), int(overrides.multistart_n))
        logger.info("Override: multi_start.n_starts = %d", int(overrides.multistart_n))

    if overrides.max_iterations is not None:
        _set_nested(cfg, (*_NLSQ_SECTION, "max_iterations"), int(overrides.max_iterations))
        logger.info("Override: nlsq.max_iterations = %d", int(overrides.max_iterations))

    if overrides.tolerance is not None:
        ftol = float(overrides.tolerance)
        _set_nested(cfg, (*_NLSQ_SECTION, "ftol"), ftol)
        _set_nested(cfg, (*_NLSQ_SECTION, "xtol"), ftol)
        # Relax gtol too: on degenerate fits trf often stops on gtol before
        # ftol/xtol, so omitting it makes --tolerance a partial no-op there.
        _set_nested(cfg, (*_NLSQ_SECTION, "gtol"), ftol)
        logger.info("Override: nlsq.ftol = nlsq.xtol = nlsq.gtol = %g", ftol)

    if overrides.verbose or overrides.quiet:
        # 0 = silent, 1 = default, 2 = chatty; quiet wins.
        v = 0 if overrides.quiet else (2 if overrides.verbose else 1)
        _set_nested(cfg, (*_NLSQ_SECTION, "verbose"), v)
