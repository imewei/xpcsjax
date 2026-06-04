"""Path validation utilities for secure file operations.

This module provides path validation functions to prevent path traversal
attacks and ensure safe file operations for save_path parameters.

Security fixes implemented as part of code review remediation (Dec 2025).
Addresses CVSS 7.5 path traversal vulnerability (VUL-001).
"""

from __future__ import annotations

from pathlib import Path

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


class PathValidationError(ValueError):
    """Raised when path validation fails due to security concerns."""

    pass


def validate_save_path(
    path: str | Path | None,
    *,
    allowed_extensions: tuple[str, ...] | None = None,
    require_parent_exists: bool = True,
    allow_absolute: bool = True,
    base_dir: Path | None = None,
) -> Path | None:
    """Validate and sanitize a file save path.

    Prevents path traversal attacks and ensures the path is safe for
    file operations.

    Parameters
    ----------
    path : str | Path | None
        Path to validate. If None, returns None.
    allowed_extensions : tuple[str, ...], optional
        Allowed file extensions (e.g., ('.png', '.pdf')).
        If None, all extensions are allowed.
    require_parent_exists : bool, default=True
        If True, validates that the parent directory exists.
    allow_absolute : bool, default=True
        If True, absolute paths are allowed.
        If False, only relative paths are allowed.
    base_dir : Path, optional
        Base directory for relative paths. If provided, the resolved
        path must be within this directory (prevents path traversal).
        Defaults to current working directory.

    Returns
    -------
    Path | None
        Validated and resolved Path object, or None if path is None.

    Raises
    ------
    PathValidationError
        If path validation fails (security concerns, bad extension, missing parent).

    Examples
    --------
    >>> validate_save_path("output/results.png")
    PosixPath('/current/dir/output/results.png')

    >>> validate_save_path("../../../etc/passwd")
    PathValidationError: Path traversal detected

    >>> validate_save_path("/tmp/test.png", allow_absolute=False)
    PathValidationError: Absolute paths not allowed
    """
    if path is None:
        return None

    # Reject null bytes before Path conversion (security: prevents null-byte injection;
    # Python 3.13+ may not raise ValueError for embedded nulls on all platforms)
    if isinstance(path, str) and "\x00" in path:
        raise PathValidationError(f"Null bytes not allowed in path: {path!r}")

    # Convert to Path object
    path = Path(path)

    # Check for path traversal by inspecting each path component.
    # Using parts instead of a raw string search avoids false positives for
    # filenames like "version..2.png" which legitimately contain "..".
    # Also split on backslashes to catch Windows-style traversal on POSIX systems
    # (e.g., "..\\..\\etc\\passwd" which Path treats as a single component on Linux).
    path_str = str(path)
    # Gather components from both the POSIX parts and any backslash-delimited segments
    raw_components = set(path.parts)
    for segment in path_str.replace("\\", "/").split("/"):
        raw_components.add(segment)
    if ".." in raw_components:
        raise PathValidationError(
            f"Path traversal detected: path contains '..': {_sanitize_log_path(path_str)}"
        )

    # Check absolute path permission
    if path.is_absolute() and not allow_absolute:
        raise PathValidationError(f"Absolute paths not allowed: {_sanitize_log_path(path_str)}")

    # Resolve the path (normalize)
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir).resolve()

    if path.is_absolute():
        resolved_path = path.resolve()
        # For explicitly allowed absolute paths, skip base_dir containment check
        # The ".." check above already prevents traversal attacks
    else:
        resolved_path = (base_dir / path).resolve()
        # For relative paths, verify resolved path is within base_dir
        # (prevents traversal via symlinks in relative paths)
        try:
            resolved_path.relative_to(base_dir)
        except ValueError as e:
            # Path is outside base_dir
            raise PathValidationError(
                f"Path resolves outside allowed directory: "
                f"{_sanitize_log_path(str(resolved_path))} is not within "
                f"{_sanitize_log_path(str(base_dir))}"
            ) from e

    # Reject paths with no filename component (e.g., root "/" or bare directory paths)
    if not resolved_path.name:
        raise PathValidationError(
            f"Path resolves to a root or directory-only path "
            f"(no filename): {_sanitize_log_path(path_str)}"
        )

    # Check extension
    if allowed_extensions is not None:
        suffix = resolved_path.suffix.lower()
        if suffix not in allowed_extensions:
            raise PathValidationError(
                f"Invalid file extension '{suffix}'. Allowed: {', '.join(allowed_extensions)}"
            )

    # Check parent directory exists
    if require_parent_exists:
        parent = resolved_path.parent
        if not parent.exists():
            raise PathValidationError(
                f"Parent directory does not exist: {_sanitize_log_path(str(parent))}"
            )
        if not parent.is_dir():
            raise PathValidationError(
                f"Parent path is not a directory: {_sanitize_log_path(str(parent))}"
            )

    return resolved_path


def validate_plot_save_path(
    path: str | Path | None,
    *,
    require_parent_exists: bool = True,
) -> Path | None:
    """Validate a save path for plot files.

    Convenience wrapper for validate_save_path with plot-specific defaults.

    Parameters
    ----------
    path : str | Path | None
        Path to validate.
    require_parent_exists : bool, default=True
        If True, validates that the parent directory exists.

    Returns
    -------
    Path | None
        Validated Path object or None.

    Raises
    ------
    PathValidationError
        If path validation fails or extension is not a valid image format.

    Examples
    --------
    >>> validate_plot_save_path("results/trace_plot.png")
    PosixPath('/current/dir/results/trace_plot.png')
    """
    # Common plot file extensions
    allowed_extensions = (
        ".png",
        ".pdf",
        ".svg",
        ".eps",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
    )

    return validate_save_path(
        path,
        allowed_extensions=allowed_extensions,
        require_parent_exists=require_parent_exists,
        allow_absolute=True,
    )


def _sanitize_log_path(path: str, max_length: int = 50) -> str:
    """Sanitize path for logging to prevent log injection.

    Parameters
    ----------
    path : str
        Path string to sanitize.
    max_length : int
        Maximum length of returned string.

    Returns
    -------
    str
        Sanitized path safe for logging.
    """
    # Remove potentially dangerous characters for log injection
    sanitized = path.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    # Truncate if too long (hide potentially sensitive deep paths)
    if len(sanitized) > max_length:
        # Show beginning and end
        return f"{sanitized[:20]}...{sanitized[-20:]}"

    return sanitized


def get_safe_output_dir(
    output_dir: str | Path | None = None,
    default_subdir: str = "xpcsjax_output",
) -> Path:
    """Get a safe output directory, creating it if necessary.

    Parameters
    ----------
    output_dir : str | Path | None
        Requested output directory. If None, uses cwd/default_subdir.
    default_subdir : str
        Default subdirectory name if output_dir is None.

    Returns
    -------
    Path
        Validated and existing output directory.

    Raises
    ------
    PathValidationError
        If the path is invalid or unsafe.
    OSError
        If directory cannot be created (permission denied, disk full, etc.).
    """
    if output_dir is None:
        output_dir = Path.cwd() / default_subdir
    else:
        output_dir = Path(output_dir)

    # Validate path doesn't contain traversal (component-level check,
    # matching validate_save_path to avoid false positives like "version..2")
    path_str = str(output_dir)
    raw_components = set(output_dir.parts)
    for segment in path_str.replace("\\", "/").split("/"):
        raw_components.add(segment)
    if ".." in raw_components:
        raise PathValidationError(
            f"Path traversal detected in output directory: {_sanitize_log_path(path_str)}"
        )

    # Resolve and create if needed
    resolved = output_dir.resolve()

    if not resolved.exists():
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            logger.debug("Created output directory: %s", _sanitize_log_path(str(resolved)))
        except OSError as e:
            raise OSError(
                f"Cannot create output directory: {_sanitize_log_path(str(resolved))}"
            ) from e

    if not resolved.is_dir():
        raise PathValidationError(
            f"Output path exists but is not a directory: {_sanitize_log_path(str(resolved))}"
        )

    return resolved
