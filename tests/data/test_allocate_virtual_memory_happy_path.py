"""Regression test for the mmap-backed virtual-memory allocator's success path.

Before this fix, _allocate_virtual_memory always raised AttributeError:
buffer._xpcsjax_mmap = mm on a plain numpy.ndarray (no __dict__) failed on
every call, so the documented >RAM-dataset fallback was completely
non-functional. tests/data/test_memory_manager_logging.py only exercises the
*failure* paths (mmap.mmap patched to raise) -- this test is the missing
happy-path check: allocate, write, read back, and confirm the buffer behaves
like a normal writable ndarray.
"""

import numpy as np

from xpcsjax.data.memory_manager import AdvancedMemoryManager


def test_allocate_virtual_memory_returns_usable_buffer(tmp_path):
    manager = AdvancedMemoryManager(
        config={
            "memory": {
                "enable_monitoring": False,
                "virtual_memory_path": str(tmp_path / "xpcsjax_vm"),
            }
        }
    )
    try:
        n = 1000
        buf = manager._allocate_virtual_memory(size=n, dtype=np.float64)  # noqa: SLF001

        assert buf.shape == (n,)
        assert buf.dtype == np.float64

        # A plain view (no __dict__) would raise AttributeError constructing
        # this buffer at all; confirm it's genuinely writable, not read-only.
        buf[:] = np.arange(n, dtype=np.float64)
        np.testing.assert_array_equal(buf, np.arange(n, dtype=np.float64))
        buf[0] = 42.0
        assert buf[0] == 42.0
    finally:
        manager.shutdown()
