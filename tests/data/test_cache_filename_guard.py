"""Cross-platform safety of the cache-filename guard.

The original guard tested ``os.sep in name`` only, so on Windows a ``/`` (a valid
separator there) or a ``C:`` drive specifier slipped through. ``_assert_safe_cache_filename``
rejects every directory/traversal/drive token regardless of the host platform.
"""

from __future__ import annotations

import pytest

from xpcsjax.data.xpcs_loader import _assert_safe_cache_filename


@pytest.mark.parametrize(
    "bad",
    [
        "../escape.npz",
        "sub/dir.npz",  # POSIX separator
        "sub\\dir.npz",  # Windows separator
        "C:evil.npz",  # Windows drive specifier
        "a:stream.npz",  # NTFS alternate data stream
        "x\x00.npz",  # null byte
    ],
)
def test_rejects_unsafe_cache_filenames(bad):
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        _assert_safe_cache_filename(bad)


def test_accepts_plain_filename():
    # The real template output is a bare filename — must pass on every platform.
    assert _assert_safe_cache_filename("cached_c2_frames_1_8000.npz") is None
