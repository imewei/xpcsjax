xpcsjax.utils
=============

Logging primitives, async I/O helpers, and output-path validation shared
across the CLI, GUI, and service layers.

.. currentmodule:: xpcsjax.utils

Package surface
----------------

.. automodule:: xpcsjax.utils
   :members:

All ten logging (``get_logger``, ``configure_logging``, ``with_context``,
``log_performance``, ``log_calls``, ``log_operation``) and path-validation
(``PathValidationError``, ``validate_save_path``, ``validate_plot_save_path``,
``get_safe_output_dir``) names are re-exported through ``xpcsjax.utils``'s
``__all__`` and rendered above.

Async I/O
---------

``PrefetchLoader`` and ``AsyncWriter`` are not re-exported through
``xpcsjax.utils`` — import them from ``xpcsjax.utils.async_io`` directly.

.. autoclass:: xpcsjax.utils.async_io.PrefetchLoader
   :members:

.. autoclass:: xpcsjax.utils.async_io.AsyncWriter
   :members:
