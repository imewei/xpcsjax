xpcsjax.utils
=============

Logging primitives, async I/O helpers, and output-path validation shared
across the CLI, GUI, and service layers.

.. currentmodule:: xpcsjax.utils

Package surface
----------------

.. automodule:: xpcsjax.utils
   :members:

Logging
-------

.. autofunction:: xpcsjax.utils.logging.get_logger

.. autofunction:: xpcsjax.utils.logging.configure_logging

.. autofunction:: xpcsjax.utils.logging.with_context

.. autofunction:: xpcsjax.utils.logging.log_performance

.. autofunction:: xpcsjax.utils.logging.log_calls

.. autofunction:: xpcsjax.utils.logging.log_operation

Async I/O
---------

.. autoclass:: xpcsjax.utils.async_io.PrefetchLoader
   :members:

.. autoclass:: xpcsjax.utils.async_io.AsyncWriter
   :members:

Path validation
----------------

.. autoclass:: xpcsjax.utils.path_validation.PathValidationError

.. autofunction:: xpcsjax.utils.path_validation.validate_save_path

.. autofunction:: xpcsjax.utils.path_validation.validate_plot_save_path

.. autofunction:: xpcsjax.utils.path_validation.get_safe_output_dir
