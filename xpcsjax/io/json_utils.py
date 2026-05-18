"""JSON utility functions for xpcsjax I/O operations.

This module provides helper functions for JSON serialization of numpy arrays
and other complex objects.
"""

import math
from typing import Any

import numpy as np


def _sanitize_float(v: float) -> float | str | None:
    """Convert non-finite floats to JSON-safe representations.

    JSON spec does not support NaN, Inf, or -Inf. These are converted to
    None (NaN) or string representations (Inf/-Inf) to prevent json.dump crashes.
    """
    if math.isnan(v):
        return None
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


def json_safe(value: Any) -> Any:
    """Recursively convert numpy arrays and special types to JSON-safe types.

    Parameters
    ----------
    value : Any
        Value to convert (can be nested dict, list, numpy array, etc.)

    Returns
    -------
    Any
        JSON-serializable version of the input

    Examples
    --------
    >>> json_safe(np.array([1, 2, 3]))
    [1, 2, 3]
    >>> json_safe({"arr": np.array([1.0, 2.0]), "val": np.float64(3.14)})
    {'arr': [1.0, 2.0], 'val': 3.14}
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    elif isinstance(value, np.ndarray):
        # Recurse through tolist() result to sanitize any NaN/Inf floats
        return json_safe(value.tolist())
    elif isinstance(value, (np.integer, np.floating)):
        v = value.item()
        if isinstance(v, float):
            return _sanitize_float(v)
        return v
    elif isinstance(value, (np.bool_,)):
        return bool(value)
    elif isinstance(value, float):
        return _sanitize_float(value)
    elif hasattr(value, "tolist"):
        # Recurse through json_safe so that custom array-like objects whose
        # tolist() returns floats containing NaN/Inf are properly sanitized.
        return json_safe(value.tolist())
    else:
        return value


def json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy arrays and other objects.

    Use as the `default` argument to json.dump/dumps.

    Parameters
    ----------
    obj : Any
        Object to serialize

    Returns
    -------
    Any
        JSON-serializable version of the object

    Raises
    ------
    TypeError
        If object cannot be serialized (will be converted to string)

    Examples
    --------
    >>> import json
    >>> json.dumps({"arr": np.array([1, 2, 3])}, default=json_serializer)
    '{"arr": [1, 2, 3]}'
    """
    if isinstance(obj, np.ndarray):
        # Recurse through json_safe so NaN/Inf floats inside the array are
        # sanitized (tolist() converts np.nan → Python float nan, which the
        # stock json encoder would either reject or emit as the invalid token NaN).
        return json_safe(obj.tolist())
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        return _sanitize_float(v)
    elif isinstance(obj, float):
        return _sanitize_float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, "tolist"):
        # Same sanitization for other array-like objects.
        return json_safe(obj.tolist())
    else:
        return str(obj)
