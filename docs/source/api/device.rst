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

.. autofunction:: xpcsjax.device.cpu.detect_cpu_info

.. autofunction:: xpcsjax.device.cpu.configure_cpu_hpc

.. autofunction:: xpcsjax.device.cpu.configure_cpu_threading

.. autofunction:: xpcsjax.device.cpu.get_optimal_batch_size

.. autofunction:: xpcsjax.device.cpu.benchmark_cpu_performance

Hardware configuration
-----------------------

Used internally for NLSQ thread and memory budget decisions.

.. autoclass:: xpcsjax.device.config.HardwareConfig
   :members:

.. autofunction:: xpcsjax.device.config.detect_hardware
