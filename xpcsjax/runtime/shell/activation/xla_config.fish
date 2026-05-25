#!/usr/bin/env fish
# XLA_FLAGS auto-configuration for xpcsjax (fish shell).
#
# Usage:
#   source xla_config.fish [mode]
#
# Modes:
#   auto - Auto-detect based on physical CPU cores (default)
#   nlsq - Single XLA device (NLSQ-only — xpcsjax has no other path)
#   <N>  - Explicit XLA device count

function _xpcsjax_resolve_mode_file
    if set -q VIRTUAL_ENV
        echo "$VIRTUAL_ENV/etc/xpcsjax/xla_mode"
    else if set -q CONDA_PREFIX
        echo "$CONDA_PREFIX/etc/xpcsjax/xla_mode"
    else
        set -l xdg_config (set -q XDG_CONFIG_HOME; and echo $XDG_CONFIG_HOME; or echo "$HOME/.config")
        echo "$xdg_config/xpcsjax/xla_mode"
    end
end

set -g _XPCSJAX_XLA_LEGACY_FILE "$HOME/.xpcsjax_xla_mode"

function _xpcsjax_get_cpu_count
    if command -v nproc > /dev/null 2>&1
        nproc
    else if command -v sysctl > /dev/null 2>&1
        sysctl -n hw.physicalcpu 2>/dev/null; or sysctl -n hw.ncpu 2>/dev/null; or echo 4
    else
        echo 4
    end
end

function _xpcsjax_configure_xla
    set -l mode $argv[1]
    test -z "$mode"; and set mode "auto"

    set -l device_count

    switch $mode
        case nlsq
            set device_count 1
        case auto
            set -l cpu_count (_xpcsjax_get_cpu_count)
            if test $cpu_count -lt 4
                set device_count $cpu_count
            else
                set device_count 4
            end
        case '*'
            if string match -qr '^\d+$' $mode
                set device_count $mode
            else
                echo "Unknown XLA mode: $mode" >&2
                return 1
            end
    end

    if set -q XPCSJAX_PRESERVE_XLA_FLAGS; and set -q XLA_FLAGS
        return 0
    end

    set -l new_flag "--xla_force_host_platform_device_count=$device_count"

    if not set -q XLA_FLAGS
        set -gx XLA_FLAGS $new_flag
    else if not string match -q "*xla_force_host_platform_device_count*" $XLA_FLAGS
        set -gx XLA_FLAGS "$XLA_FLAGS $new_flag"
    end

    if not set -q JAX_PLATFORMS
        set -gx JAX_PLATFORMS cpu
    end
end

function _xpcsjax_save_xla_mode
    set -l mode $argv[1]
    set -l mode_file (_xpcsjax_resolve_mode_file)
    mkdir -p (dirname $mode_file)
    echo $mode > $mode_file

    if test -f $_XPCSJAX_XLA_LEGACY_FILE
        rm -f $_XPCSJAX_XLA_LEGACY_FILE
    end
end

function _xpcsjax_load_xla_mode
    set -l mode_file (_xpcsjax_resolve_mode_file)
    if test -f $mode_file
        cat $mode_file
    else if test -f $_XPCSJAX_XLA_LEGACY_FILE
        cat $_XPCSJAX_XLA_LEGACY_FILE
    else
        echo "auto"
    end
end

function _xpcsjax_xla_setup
    set -l mode

    if test (count $argv) -gt 0
        set mode $argv[1]
        _xpcsjax_save_xla_mode $mode
    else
        set mode (_xpcsjax_load_xla_mode)
    end

    _xpcsjax_configure_xla $mode
end

if test (count $argv) -gt 0
    _xpcsjax_xla_setup $argv[1]
else
    _xpcsjax_xla_setup
end
