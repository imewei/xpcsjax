"""F6: NLSQConfig.validate() upper-bound guard for chunk-size fields.

A pathologically large chunk_size (e.g. 10_000_000_001) must be rejected by
validate() to prevent config-driven DoS (unbounded allocation).  The ceiling
is MAX_CHUNK_SIZE (100_000_000 points), justified by typical XPCS dataset
sizes being well below 10 M points.

Regression contract:
- Values above MAX_CHUNK_SIZE produce a validate() error string.
- Normal values (e.g. 100_000) are accepted.
- The existing <= 0 checks still fire.
"""

from __future__ import annotations

from xpcsjax.optimization.nlsq.config import MAX_CHUNK_SIZE, NLSQConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ABOVE_CEILING = MAX_CHUNK_SIZE + 1
NORMAL = 100_000
BELOW_ZERO = 0


# ---------------------------------------------------------------------------
# streaming_chunk_size
# ---------------------------------------------------------------------------


def test_streaming_chunk_size_above_ceiling_is_rejected() -> None:
    cfg = NLSQConfig(streaming_chunk_size=ABOVE_CEILING)
    errors = cfg.validate()
    assert any("streaming_chunk_size" in e for e in errors), (
        f"expected error for streaming_chunk_size={ABOVE_CEILING}, got: {errors}"
    )


def test_streaming_chunk_size_normal_is_accepted() -> None:
    cfg = NLSQConfig(streaming_chunk_size=NORMAL)
    errors = cfg.validate()
    assert not any("streaming_chunk_size" in e for e in errors), (
        f"unexpected error for streaming_chunk_size={NORMAL}: {errors}"
    )


def test_streaming_chunk_size_zero_still_rejected() -> None:
    cfg = NLSQConfig(streaming_chunk_size=BELOW_ZERO)
    errors = cfg.validate()
    assert any("streaming_chunk_size" in e for e in errors), (
        f"expected lower-bound error for streaming_chunk_size=0, got: {errors}"
    )


# ---------------------------------------------------------------------------
# target_chunk_size
# ---------------------------------------------------------------------------


def test_target_chunk_size_above_ceiling_is_rejected() -> None:
    cfg = NLSQConfig(target_chunk_size=ABOVE_CEILING)
    errors = cfg.validate()
    assert any("target_chunk_size" in e for e in errors), (
        f"expected error for target_chunk_size={ABOVE_CEILING}, got: {errors}"
    )


def test_target_chunk_size_normal_is_accepted() -> None:
    cfg = NLSQConfig(target_chunk_size=NORMAL)
    errors = cfg.validate()
    assert not any("target_chunk_size" in e for e in errors), (
        f"unexpected error for target_chunk_size={NORMAL}: {errors}"
    )


def test_target_chunk_size_zero_still_rejected() -> None:
    cfg = NLSQConfig(target_chunk_size=BELOW_ZERO)
    errors = cfg.validate()
    assert any("target_chunk_size" in e for e in errors), (
        f"expected lower-bound error for target_chunk_size=0, got: {errors}"
    )


# ---------------------------------------------------------------------------
# hybrid_chunk_size
# ---------------------------------------------------------------------------


def test_hybrid_chunk_size_above_ceiling_is_rejected() -> None:
    cfg = NLSQConfig(hybrid_chunk_size=ABOVE_CEILING)
    errors = cfg.validate()
    assert any("hybrid_chunk_size" in e for e in errors), (
        f"expected error for hybrid_chunk_size={ABOVE_CEILING}, got: {errors}"
    )


def test_hybrid_chunk_size_normal_is_accepted() -> None:
    cfg = NLSQConfig(hybrid_chunk_size=NORMAL)
    errors = cfg.validate()
    assert not any("hybrid_chunk_size" in e for e in errors), (
        f"unexpected error for hybrid_chunk_size={NORMAL}: {errors}"
    )


def test_hybrid_chunk_size_zero_still_rejected() -> None:
    cfg = NLSQConfig(hybrid_chunk_size=BELOW_ZERO)
    errors = cfg.validate()
    assert any("hybrid_chunk_size" in e for e in errors), (
        f"expected lower-bound error for hybrid_chunk_size=0, got: {errors}"
    )


# ---------------------------------------------------------------------------
# cmaes_data_chunk_size  (None = auto is allowed; only positive ints are bounded)
# ---------------------------------------------------------------------------


def test_cmaes_data_chunk_size_above_ceiling_is_rejected() -> None:
    cfg = NLSQConfig(cmaes_data_chunk_size=ABOVE_CEILING)
    errors = cfg.validate()
    assert any("cmaes_data_chunk_size" in e for e in errors), (
        f"expected error for cmaes_data_chunk_size={ABOVE_CEILING}, got: {errors}"
    )


def test_cmaes_data_chunk_size_normal_is_accepted() -> None:
    cfg = NLSQConfig(cmaes_data_chunk_size=NORMAL)
    errors = cfg.validate()
    assert not any("cmaes_data_chunk_size" in e for e in errors), (
        f"unexpected error for cmaes_data_chunk_size={NORMAL}: {errors}"
    )


def test_cmaes_data_chunk_size_none_is_accepted() -> None:
    cfg = NLSQConfig(cmaes_data_chunk_size=None)
    errors = cfg.validate()
    assert not any("cmaes_data_chunk_size" in e for e in errors), (
        f"unexpected error for cmaes_data_chunk_size=None: {errors}"
    )


def test_cmaes_data_chunk_size_zero_still_rejected() -> None:
    cfg = NLSQConfig(cmaes_data_chunk_size=BELOW_ZERO)
    errors = cfg.validate()
    assert any("cmaes_data_chunk_size" in e for e in errors), (
        f"expected lower-bound error for cmaes_data_chunk_size=0, got: {errors}"
    )
