# ruff: noqa
# packaging/xpcsjax-gui.spec — build with: uv run pyinstaller packaging/xpcsjax-gui.spec
# Gotchas (see packaging/README.md):
#  - multiprocessing spawn: main() calls freeze_support(); the spec must bundle the
#    same interpreter so spawned workers re-exec correctly.
#  - JAX/jaxlib + PySide6 + pyqtgraph + datashader (numba) ship large data files;
#    collect them via collect_all / collect_data_files.
from PyInstaller.utils.hooks import collect_all  # noqa: E402

datas, binaries, hiddenimports = [], [], []
# The full xpcsjax runtime stack — compiled extensions + data files PyInstaller's
# static analysis misses (verified against pyproject dependencies):
for pkg in (
    "jax", "jaxlib", "xpcsjax", "PySide6", "pyqtgraph",
    "datashader", "xarray", "colorcet", "numba", "llvmlite", "nlsq", "evosax",
    "jaxopt", "interpax", "h5py", "sklearn", "scipy",
    "matplotlib",  # worker-rendered publication figures ship mpl-data
    "numpy",        # direct runtime dep; pulled transitively by jax/scipy but
                    # listed explicitly so the drift guard requires no special-case
):  # xarray/colorcet are datashader's array backend + colormap data (ship data files);
    # they are direct `viz-fast` deps, so the drift-guard test below requires them here.
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

a = Analysis(["packaging/launch_gui.py"], datas=datas, binaries=binaries, hiddenimports=hiddenimports)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, name="xpcsjax-gui", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="xpcsjax-gui")
