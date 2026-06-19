# xpcsjax-gui — PyInstaller packaging notes

## Building the app bundle

The binary build is **per-OS** — you must build on the target platform:

```bash
# Install the packaging extra first
uv pip install -e ".[packaging]"

# Build (one-dir bundle recommended over one-file; see below)
uv run pyinstaller packaging/xpcsjax-gui.spec --noconfirm
```

The output lands in `dist/xpcsjax-gui/` (one-dir bundle).

## Why one-dir, not one-file?

One-file bundles re-extract the entire app into a temp directory on every
process launch, including every `multiprocessing` **spawn** worker. With the
JAX/numba/datashader data footprint this re-extraction is slow (several
seconds per worker) and can race on parallel worker launches. One-dir bundles
re-exec the already-extracted files from disk — spawn workers start cleanly
and quickly.

## spawn / freeze_support requirement

xpcsjax uses `multiprocessing` for its HYBRID_STREAMING worker pool. In a
frozen app the default spawn start method causes every spawned worker to
re-run the entry point. The `multiprocessing.freeze_support()` call at the
top of `xpcsjax.gui.app.main()` short-circuits this: workers detect they are
a frozen subprocess and go directly to their function, skipping the GUI.

**Without `freeze_support()` the frozen GUI will re-open a new window for
every spawned worker.** The call is a no-op when running from source.

## Large data collection (JAX / numba / datashader)

JAX, jaxlib, numba, llvmlite, PySide6, pyqtgraph, datashader, and matplotlib
all ship compiled extensions and/or large data files that PyInstaller's static
import analysis misses. The spec uses `collect_all()` for every package in the
runtime stack to ensure these are bundled. Expect the `dist/` directory to be
several hundred MB.

## Build environment

- Build on the **same OS** and **same Python version** as the target machine.
- Do **not** build inside a virtual machine on a different host OS and expect
  the binary to run natively — shared libraries are platform-specific.
- The `make package` Makefile target runs the build; it is not wired into
  `make verify` or CI (the unit test suite only validates freeze-safety
  properties, not the actual binary output).

## CI / maintainer-only

The binary build (`make package`) is intended for maintainer machines and
dedicated release CI runners that have the full platform SDK available. It is
**not** run in the standard unit-test pipeline (`make test` / `make verify`).
The `tests/gui/test_freeze_safety.py` suite validates the source-level
freeze-safety contract (freeze_support placement, console-script registration,
spec drift guard) without invoking PyInstaller.
