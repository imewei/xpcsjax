"""Tests for heterodyne joint multistart wiring (Phase 1)."""
from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_multistart import build_multistart_config


def test_build_multistart_config_reads_nested_keys() -> None:
    ms_dict = {
        "enable": True,
        "n_starts": 7,
        "seed": 99,
        "sampling_strategy": "latin_hypercube",
        "n_workers": 4,
        "use_screening": False,
        "screen_keep_fraction": 0.3,
        "refine_top_k": 2,
        "refinement_ftol": 1e-10,
        "degeneracy_threshold": 0.25,
    }
    cfg = build_multistart_config(ms_dict)
    assert cfg.enable is True
    assert cfg.n_starts == 7
    assert cfg.seed == 99
    assert cfg.use_screening is False
    assert cfg.screen_keep_fraction == 0.3
    assert cfg.refine_top_k == 2
    assert cfg.degeneracy_threshold == 0.25
    # Heterodyne worker closes over a JAX model -> not process-picklable.
    # n_workers MUST be clamped to 1 (sequential) regardless of config.
    assert cfg.n_workers == 1


def test_build_multistart_config_defaults_on_empty() -> None:
    cfg = build_multistart_config({})
    assert cfg.enable is False
    assert cfg.n_starts == 10
    assert cfg.seed == 42
    assert cfg.n_workers == 1
