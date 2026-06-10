Decision record: CPU-only execution, GPU deferred
=================================================

:Status: Accepted for v0.1 (reviewed 2026-06-09).
:Scope: Applies to the NLSQ fit path — the supported workload of v0.1.
:Supersedes: The inline 2026-03 rationale that previously lived in
   ``xpcsjax/device/cpu.py`` (``_configure_jax_cpu``). Two of its premises were
   factually wrong and are corrected below.

Decision
--------

**xpcsjax v0.1 runs on CPU, and a GPU is not required.** JAX is pinned to the
CPU backend at import time (``xpcsjax/__init__.py`` sets
``JAX_PLATFORMS=cpu``), and ``NLSQ_SKIP_GPU_CHECK=1`` silences the upstream
NLSQ GPU probe.

Implementing a first-class GPU path is **deferred, not because it is hard, but
because it does not help the workloads xpcsjax actually runs.** The decision of
*whether* GPU is worth wiring in is empirical and gated on dataset size — see
`When to revisit`_. It is not blocked behind any solver rewrite.

Context
-------

XPCS NLSQ fitting here is float64-mandatory: physical parameters span 6–7
orders of magnitude, so ``JAX_ENABLE_X64=1`` is non-negotiable (see
:doc:`/advanced/jax_environment`). The natural question is whether offloading
the fit to a GPU would speed it up enough to be worth supporting.

Reasoning
---------

The decision rests on four facts, in order of weight.

1. The bottleneck is compilation, not compute (decisive)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profiling on ``main`` (2026-06-05, HYBRID_STREAMING, C044 ≈ 1.55 M points)
breaks the fit's wall-clock time down as:

* **≈ 47 %** one-time JAX/XLA **cold compilation** (62 distinct ``pjit``
  signatures; 0 recompiles across warm iterations);
* a large share in the **host-side one-time index build**
  (``_precompute_chunk_metadata`` / ``_compute_flat_indices`` in
  ``strategies/residual.py``) — integer math, not floating-point;
* **≈ 1.6 %** in the **warm numeric loop** — the *only* part a GPU
  accelerates. That loop already runs at ≈ 43.5 M points/s and is a known
  performance plateau ("do not touch" — regression trap).

A GPU accelerates the 1.6 %. By Amdahl's law the maximum achievable wall-clock
improvement is therefore ≈ ``1 / (1 - 0.016) ≈ 1.016×`` — about 2 %, *even with
an infinitely fast kernel*. GPU cold-compilation is typically **slower** than
CPU, so the realistic net effect on the current workload is neutral-to-negative.

2. Float64 on consumer GPUs erases the compute advantage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consumer GPUs run float64 at a 1:64 throughput ratio. An RTX 4090 delivers
≈ 1.3 TFLOPS float64 — comparable to a 20-core CPU. NLSQ's advertised
"20–100× speedup" is a **float32** figure and does not apply to this
float64, element-wise-dominated workload. A clear GPU win needs datacenter
hardware (A100/H100, 1:2 float64 ratio), not the laptop-class GPU present on
the maintainer machine.

3. There is no rewrite barrier — the solver is already GPU-ready
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The upstream NLSQ trust-region (Levenberg–Marquardt) solve is **pure JAX**:
``nlsq.common_jax._solve_lsq_trust_region_jax_impl`` is built from
``lax.while_loop`` / ``lax.cond`` under ``@jit`` and stays entirely
on-device. NLSQ 0.6.12 ships **no compiled extensions** (no ``.so``, no
``.pyx``). Enabling GPU is therefore an *install + device-flag* change, not an
optimizer rewrite.

This makes the decision cheap to revisit empirically (point 4) and means the
"deferral" carries no hidden engineering debt.

4. Testing GPU is one command, so the bar is "measure, then decide"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because ``xpcsjax/__init__.py`` uses ``os.environ.setdefault("JAX_PLATFORMS",
"cpu")``, an exported ``JAX_PLATFORMS=cuda`` already overrides the pin with no
code change. With a CUDA-enabled ``jaxlib`` installed, a GPU A/B run against a
real dataset is immediate. The decision is data-gated, not speculative.

.. _superseded-reasoning:

Superseded (incorrect) reasoning
--------------------------------

The original 2026-03 rationale justified the CPU pin partly on claims that do
**not** hold and must not be propagated:

.. admonition:: Corrected
   :class: warning

   * **"PCIe round-trips forced by the NLSQ C extension (~70 ms/iteration)."**
     False. NLSQ has no C extension; the LM solve is pure JAX and never leaves
     the device per iteration.
   * **"Viable only with … a jaxopt-based optimizer rewrite."** False. No
     rewrite is required (the solver is already JAX-native). ``jaxopt`` is, in
     any case, *already* a dependency — it backs the L2 hierarchical optimizer
     in ``optimization/nlsq/hierarchical.py``.

   The **valid** parts of the original rationale — float64's 1:64 penalty on
   consumer GPUs and the CPU-only XLA flags — are retained above. The
   *governing* reason, however, is the compile-dominated profile (point 1),
   which is independent of hardware.

Consequences
------------

* CPU is the supported, correct target for v0.1. A multi-core CPU — ideally a
  many-core HPC node (the ``xpcsjax/device/cpu.py`` module is NUMA-aware and
  tuned for 36/128-core Intel Xeon / AMD EPYC) — is where the package is meant
  to run.
* Installing a GPU does nothing on its own: JAX is pinned to CPU before it can
  enumerate a device.
* Performance effort for v0.1 should target the **47 %**, not the **1.6 %** —
  i.e. reduce the 62 distinct ``pjit`` compile signatures and move the
  one-time host-computable index math off the critical path (bit-identical).
  The JAX persistent compilation cache is *architecturally* blocked for this
  fit path (data baked into the JIT closure → 0 cache entries written), so it
  is not a lever either.

When to revisit
---------------

Reconsider a first-class GPU path only when an A/B benchmark shows the **warm
numeric loop dominating** wall-clock time — which requires *all* of:

#. workloads large enough (the 23 M+ point HYBRID_STREAMING regime, run
   repeatedly so compilation amortizes) that the numeric fraction overtakes
   compile + host-index cost; **and**
#. datacenter-class GPUs (A100/H100, 1:2 float64), not consumer cards; **and**
#. a measured GPU-vs-CPU win on real datasets, not NLSQ's float32 headline
   number.

If those hold, "implement GPU" reduces to: drop the hard CPU pin, add a
device-selection config flag, and gate the float64 datacenter path. Days of
work, not a rewrite.
