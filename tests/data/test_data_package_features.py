"""Regression guard for xpcsjax.data's import contract.

A dangling name in one of xpcsjax/data/__init__.py's imports (e.g. a name
deleted from the submodule it imports from) used to silently flip the
corresponding HAS_* flag to False and hide the whole feature instead of
raising -- see PR #68 (commit f244c3f), where a stale
`create_default_preprocessing_config`/`preprocess_xpcs_data` import broke
HAS_PREPROCESSING this exact way with no test catching it. The try/except
ImportError "maybe-present" shims (and their HAS_* flags) were removed in
the 2026-09-15 review (finding B3): every submodule is a hard/in-tree
dependency, so a broken import here now raises at package-import time
instead of degrading silently -- this test pins the surviving static
``__features__`` contract.
"""

from __future__ import annotations

import xpcsjax.data as data


def test_preprocessing_symbols_are_exported() -> None:
    assert "PreprocessingPipeline" in data.__all__
    assert data.PreprocessingPipeline is not None


def test_feature_flags_are_gone() -> None:
    # Every data submodule is a hard/in-tree dependency; the old __features__
    # dict only ever reported True and measured nothing, so it was removed.
    assert not hasattr(data, "__features__")
    assert "features" not in data.get_data_module_info()


def test_has_flags_removed() -> None:
    """The per-feature HAS_* shims (and the loader-error escape hatch) are gone."""
    for name in (
        "HAS_XPCS_LOADER",
        "HAS_VALIDATION",
        "HAS_PHI_FILTERING",
        "HAS_ANGLE_FILTERING",
        "HAS_PREPROCESSING",
        "HAS_OPTIMIZATION",
        "HAS_VALIDATORS",
        "_loader_error",
    ):
        assert not hasattr(data, name), f"{name} should have been removed (finding B3)"
