xpcsjax.utils
=============

Logging primitives and output-path validation shared across the CLI, GUI,
and service layers.

.. currentmodule:: xpcsjax.utils

Package surface
----------------

.. automodule:: xpcsjax.utils
   :members:

All five logging (``get_logger``, ``configure_logging``, ``with_context``,
``log_performance``, ``log_calls``) and path-validation
(``PathValidationError``, ``validate_save_path``, ``validate_plot_save_path``,
``get_safe_output_dir``) names are re-exported through ``xpcsjax.utils``'s
``__all__`` and rendered above.
