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

``detect_cpu_info``, ``configure_cpu_hpc``, ``get_optimal_batch_size``, and
``benchmark_cpu_performance`` are re-exported through ``xpcsjax.device``'s
``__all__`` and already rendered above by the package-level ``automodule``.
``configure_cpu_threading`` is not re-exported, so it needs its own entry:

.. autofunction:: xpcsjax.device.cpu.configure_cpu_threading

Hardware configuration
-----------------------

Used internally for NLSQ thread and memory budget decisions.

.. autoclass:: xpcsjax.device.config.HardwareConfig
   :members:

.. autofunction:: xpcsjax.device.config.detect_hardware
