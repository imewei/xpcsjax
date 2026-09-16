"""Small shared helpers deduplicated across ``wrapper.py`` and ``adapter.py``.

``get_physical_param_names`` used to be hardcoded three times (``wrapper.py``,
``adapter.py``, ``core.py``) even though
:mod:`xpcsjax.config.parameter_registry` already owns parameter names/bounds
as the single source of truth (see root ``CLAUDE.md``). ``extract_nlsq_settings``
used to be a byte-identical copy on ``NLSQWrapper`` and ``NLSQAdapter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from xpcsjax.config.parameter_registry import AnalysisMode

__all__ = ["extract_nlsq_settings", "get_physical_param_names"]


def get_physical_param_names(analysis_mode: AnalysisMode) -> list[str]:
    """Physical parameter names for an analysis mode.

    Delegates to :func:`xpcsjax.config.parameter_registry.get_param_names`
    instead of duplicating its name lists a third time.
    """
    from xpcsjax.config.parameter_registry import get_param_names

    return get_param_names(analysis_mode)


def extract_nlsq_settings(config: Any) -> dict[str, Any]:
    """Return the ``optimization.nlsq`` settings sub-tree from ``config`` (if any).

    Accepts either a ``ConfigManager``-like object (``.config`` dict attr) or
    a plain dict. ``or {}`` (not ``.get(k, {})``) so a present-but-null YAML
    section (``optimization:`` / ``nlsq:`` with no body) degrades to defaults
    instead of raising ``AttributeError`` on the next ``.get()``.
    """
    config_dict = None
    if hasattr(config, "config") and isinstance(config.config, dict):
        config_dict = config.config
    elif isinstance(config, dict):
        config_dict = config

    if not config_dict:
        return {}

    nlsq_settings = (config_dict.get("optimization") or {}).get("nlsq") or {}
    return cast(dict[str, Any], nlsq_settings)
