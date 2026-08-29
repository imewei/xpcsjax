xpcsjax.device
===============

CPU-only HPC device detection and thread/NUMA optimization. GPU support was
removed for v0.1 — every function here targets CPU-only environments.

.. currentmodule:: xpcsjax.device

Package surface
----------------

.. automodule:: xpcsjax.device
   :members:

CPU detection and configuration
--------------------------------

``detect_cpu_info`` and ``configure_cpu_hpc`` are re-exported through
``xpcsjax.device``'s ``__all__`` and already rendered above by the
package-level ``automodule``. ``configure_cpu_threading`` is not
re-exported, so it needs its own entry:

.. autofunction:: xpcsjax.device.cpu.configure_cpu_threading
