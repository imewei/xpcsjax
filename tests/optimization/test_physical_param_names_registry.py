"""C1: ``NLSQWrapper``, ``NLSQAdapter``, and ``core.py`` used to each carry
their own hardcoded physical-parameter-name list even though
``xpcsjax.config.parameter_registry`` already owns parameter names as the
single source of truth (root CLAUDE.md). All three now delegate to
:mod:`xpcsjax.optimization.nlsq.nlsq_settings`, which itself delegates to the
registry -- this pins that they still agree, and that ``extract_nlsq_settings``
behaves identically for both callers.
"""

from __future__ import annotations

import pytest

from xpcsjax.config.parameter_registry import AnalysisMode, get_param_names
from xpcsjax.optimization.nlsq.adapter import NLSQAdapter
from xpcsjax.optimization.nlsq.core import _get_physical_param_names as core_names
from xpcsjax.optimization.nlsq.nlsq_settings import extract_nlsq_settings
from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper


@pytest.mark.parametrize(
    "mode",
    [AnalysisMode.STATIC_ANISOTROPIC, AnalysisMode.STATIC_ISOTROPIC, AnalysisMode.LAMINAR_FLOW],
)
def test_all_three_sites_agree_with_the_registry(mode):
    expected = get_param_names(mode)
    assert NLSQWrapper._get_physical_param_names(mode) == expected
    assert NLSQAdapter._get_physical_param_names(mode) == expected
    assert core_names(mode) == expected


def test_unknown_mode_raises_on_all_three_sites():
    for fn in (
        NLSQWrapper._get_physical_param_names,
        NLSQAdapter._get_physical_param_names,
        core_names,
    ):
        with pytest.raises(ValueError):
            fn("not_a_real_mode")


class _Cfg:
    def __init__(self, config):
        self.config = config


def test_extract_nlsq_settings_null_section_degrades_to_empty_dict():
    cfg = _Cfg({"optimization": {"nlsq": None}})
    assert NLSQWrapper._extract_nlsq_settings(cfg) == {}
    assert NLSQAdapter._extract_nlsq_settings(cfg) == {}
    assert extract_nlsq_settings(cfg) == {}


def test_extract_nlsq_settings_reads_through_nested_dict():
    cfg = _Cfg({"optimization": {"nlsq": {"loss": "soft_l1"}}})
    assert NLSQWrapper._extract_nlsq_settings(cfg) == {"loss": "soft_l1"}
    assert NLSQAdapter._extract_nlsq_settings(cfg) == {"loss": "soft_l1"}
