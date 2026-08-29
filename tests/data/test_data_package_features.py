"""Regression guard for xpcsjax.data's feature-flag/import contract.

A dangling name in one of xpcsjax/data/__init__.py's try/except import
blocks (e.g. a name deleted from the submodule it imports from) silently
flips the corresponding HAS_* flag to False and hides the whole feature
instead of raising -- see PR #68 (commit f244c3f), where a stale
`create_default_preprocessing_config`/`preprocess_xpcs_data` import broke
HAS_PREPROCESSING this exact way with no test catching it.
"""

from __future__ import annotations

import xpcsjax.data as data


def test_has_preprocessing_is_true() -> None:
    assert data.HAS_PREPROCESSING is True


def test_preprocessing_symbols_are_exported() -> None:
    assert "PreprocessingPipeline" in data.__all__
    assert data.PreprocessingPipeline is not None


def test_all_feature_flags_are_true() -> None:
    # Every HAS_* flag in xpcsjax.data guards a real submodule that always
    # ships in-package (no optional extras) -- a False here means an import
    # inside the corresponding try/except block is broken, not that a
    # genuinely-optional dependency is missing.
    assert data.__features__ == {
        "xpcs_loader": True,
        "validation": True,
        "phi_filtering": True,
        "angle_filtering": True,
        "preprocessing": True,
        "optimization": True,
        "validators": True,
        "yaml_config": True,
        "json_support": True,
    }
