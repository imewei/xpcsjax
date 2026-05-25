#!/bin/bash
# XLA_FLAGS auto-configuration for xpcsjax (bash/zsh compatible).
#
# Source this from your venv's activate script (the xpcsjax post-install
# script does this automatically).
#
# Usage:
#   source xla_config.bash [mode]
#
# Modes:
#   auto - Auto-detect based on physical CPU cores (default)
#   nlsq - Single XLA device (NLSQ-only — xpcsjax has no other path)
#   <N>  - Explicit XLA device count

_xpcsjax_resolve_mode_file() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "${VIRTUAL_ENV}/etc/xpcsjax/xla_mode"
    elif [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo "${CONDA_PREFIX}/etc/xpcsjax/xla_mode"
    else
        echo "${XDG_CONFIG_HOME:-$HOME/.config}/xpcsjax/xla_mode"
    fi
}

_XPCSJAX_XLA_LEGACY_FILE="${HOME}/.xpcsjax_xla_mode"

_xpcsjax_get_cpu_count() {
    if command -v nproc &>/dev/null; then
        nproc
    elif command -v sysctl &>/dev/null; then
        sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
    else
        echo 4
    fi
}

_xpcsjax_configure_xla() {
    local mode="${1:-auto}"
    local device_count

    case "$mode" in
        nlsq)
            # xpcsjax is NLSQ-only; single device is the standard mode.
            device_count=1
            ;;
        auto)
            local cpu_count
            cpu_count=$(_xpcsjax_get_cpu_count)
            # Match xpcsjax/__init__.py's --xla_force_host_platform_device_count=4
            # default but cap at physical cores when smaller.
            device_count=$((cpu_count < 4 ? cpu_count : 4))
            ;;
        [0-9]*)
            device_count="$mode"
            ;;
        *)
            echo "Unknown XLA mode: $mode" >&2
            return 1
            ;;
    esac

    # Honor user-set XLA_FLAGS if requested
    if [[ -n "${XPCSJAX_PRESERVE_XLA_FLAGS:-}" && -n "${XLA_FLAGS:-}" ]]; then
        return 0
    fi

    local new_flag="--xla_force_host_platform_device_count=${device_count}"

    if [[ -z "${XLA_FLAGS:-}" ]]; then
        export XLA_FLAGS="$new_flag"
    elif [[ "$XLA_FLAGS" != *"xla_force_host_platform_device_count"* ]]; then
        export XLA_FLAGS="${XLA_FLAGS} ${new_flag}"
    fi

    # CPU-only in v0.1 (per project CLAUDE.md: GPU support is v0.2+)
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
}

_xpcsjax_save_xla_mode() {
    local mode="$1"
    local mode_file
    mode_file=$(_xpcsjax_resolve_mode_file)
    mkdir -p "$(dirname "$mode_file")"
    local tmp_file
    tmp_file=$(mktemp "${mode_file}.XXXXXX")
    echo "$mode" > "$tmp_file"
    mv "$tmp_file" "$mode_file"

    if [[ -f "$_XPCSJAX_XLA_LEGACY_FILE" ]]; then
        rm -f "$_XPCSJAX_XLA_LEGACY_FILE"
    fi
}

_xpcsjax_load_xla_mode() {
    local mode_file
    mode_file=$(_xpcsjax_resolve_mode_file)
    if [[ -f "$mode_file" ]]; then
        cat "$mode_file"
    elif [[ -f "$_XPCSJAX_XLA_LEGACY_FILE" ]]; then
        cat "$_XPCSJAX_XLA_LEGACY_FILE"
    else
        echo "auto"
    fi
}

_xpcsjax_xla_setup() {
    local mode

    if [[ -n "${1:-}" ]]; then
        mode="$1"
        _xpcsjax_save_xla_mode "$mode"
    else
        mode=$(_xpcsjax_load_xla_mode)
    fi

    _xpcsjax_configure_xla "$mode"
}

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _xpcsjax_xla_setup "${1:-}"
fi
