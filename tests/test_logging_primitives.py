"""Defensive-logging contract tests for the logging primitives.

Logging is OBSERVATIONAL ONLY: a logging failure must never escape and must
never change numerical results or control flow. These tests pin that contract
for ``log_exception``, ``log_phase`` and ``log_quantile_scaling``.
"""

import logging
from unittest.mock import MagicMock

import numpy as np

import xpcsjax.utils.logging as lm
from xpcsjax.optimization.nlsq import heterodyne_logging as hl


def test_log_exception_never_raises_on_bad_context():
    logger = MagicMock(spec=logging.Logger)

    class Boom:
        def __repr__(self):
            raise RuntimeError("repr blew up")

    lm.log_exception(logger, ValueError("x"), context={"bad": Boom()})
    assert logger.error.called or logger.log.called


def test_log_quantile_scaling_never_raises_on_empty():
    hl.log_quantile_scaling(np.array([]), np.array([]))  # must not raise


def test_log_phase_never_raises_when_memory_probe_fails(monkeypatch):
    monkeypatch.setattr(
        lm, "_get_memory_gb", lambda: (_ for _ in ()).throw(RuntimeError("probe"))
    )
    with lm.log_phase("p", track_memory=True):
        pass  # must not raise
