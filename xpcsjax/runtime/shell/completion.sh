#!/bin/bash
# shellcheck disable=SC2034  # words/cword set by _init_completion convention
# Bash/zsh completion for xpcsjax CLI commands.
#
# Installation:
#   Source this file from your .bashrc or copy to /etc/bash_completion.d/.
#   The xpcsjax post-install script wires it into the venv activate.
#
# Features:
#   * Context-aware completions for --config (YAML), --mode, --output-format, etc.
#   * Config-file caching (5-minute TTL) under XDG_CACHE_HOME.

# Cache directory for completions
_XPCSJAX_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/xpcsjax"
_XPCSJAX_CACHE_TTL=300  # 5 minutes

_xpcsjax_ensure_cache() {
    [[ -d "$_XPCSJAX_CACHE_DIR" ]] || mkdir -p "$_XPCSJAX_CACHE_DIR"
}

# Fallback for _init_completion when bash-completion is not loaded
# (common in conda/mamba environments)
if ! type _init_completion &>/dev/null; then
    _init_completion() {
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
    }

    if ! type _filedir &>/dev/null; then
        _filedir() {
            if [[ "$1" == "-d" ]]; then
                mapfile -t COMPREPLY < <(compgen -d -- "${cur}")
            else
                mapfile -t COMPREPLY < <(compgen -f -- "${cur}")
            fi
        }
    fi
fi

# Cached lookup of YAML files in the cwd and obvious subdirs
_xpcsjax_get_config_files() {
    _xpcsjax_ensure_cache
    local cache_file="$_XPCSJAX_CACHE_DIR/config_files"
    local now
    now=$(date +%s)

    if [[ -f "$cache_file" ]]; then
        local cache_time
        cache_time=$(stat -f %m "$cache_file" 2>/dev/null || stat -c %Y "$cache_file" 2>/dev/null)
        if [[ $((now - cache_time)) -lt $_XPCSJAX_CACHE_TTL ]]; then
            cat "$cache_file"
            return
        fi
    fi

    {
        find . -maxdepth 2 \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "config" ]] && find config \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "configs" ]] && find configs \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
    } | sort -u | tee "$cache_file"
}

# Main xpcsjax completion
_xpcsjax() {
    local cur prev words cword
    _init_completion -s || return

    local global_opts="--config -c --mode --output -o --output-format --verbose -v --quiet -q --help --version"
    local nlsq_opts="--multistart --multistart-n --max-iterations --tolerance"
    local param_opts="--initial-D0 --initial-alpha --initial-D-offset --initial-gamma-dot-t0 --initial-gamma-dot-t-offset --initial-v-beta --initial-v0 --initial-v-offset --initial-phi0 --initial-f0 --initial-f1 --initial-f2 --initial-f3 --initial-contrast --initial-offset"
    local perf_opts="--threads --no-jit"
    local plot_opts="--plot --no-plot --plot-experimental-data --plot-simulated-data --contrast --offset-sim --save-plots --plotting-backend --parallel-plots --phi-angles --phi"
    local all_opts="${global_opts} ${nlsq_opts} ${param_opts} ${perf_opts} ${plot_opts}"

    case "$prev" in
        --config|-c)
            mapfile -t COMPREPLY < <(compgen -W "$(_xpcsjax_get_config_files)" -- "${cur}")
            return
            ;;
        --mode)
            mapfile -t COMPREPLY < <(compgen -W "static_anisotropic static_isotropic laminar_flow two_component" -- "${cur}")
            return
            ;;
        --output|-o)
            _filedir -d
            return
            ;;
        --output-format)
            mapfile -t COMPREPLY < <(compgen -W "json npz both" -- "${cur}")
            return
            ;;
        --plotting-backend)
            mapfile -t COMPREPLY < <(compgen -W "auto matplotlib datashader" -- "${cur}")
            return
            ;;
        --threads|--multistart-n|--max-iterations)
            return
            ;;
        --phi|--contrast|--offset-sim|--tolerance|\
        --initial-D0|--initial-alpha|--initial-D-offset|\
        --initial-gamma-dot-t0|--initial-gamma-dot-t-offset|\
        --initial-v-beta|--initial-v0|--initial-v-offset|--initial-phi0|\
        --initial-f0|--initial-f1|--initial-f2|--initial-f3|\
        --initial-contrast|--initial-offset)
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${all_opts}" -- "${cur}")
        return
    fi

    mapfile -t COMPREPLY < <(compgen -W "$(_xpcsjax_get_config_files) ${all_opts}" -- "${cur}")
}

# xpcsjax-config completion
_xpcsjax_config() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--mode --output -o --data -d --q --dt --time-length --overwrite --show-template --interactive --validate --help"

    case "$prev" in
        --output|-o|--data|-d|--validate)
            _filedir
            return
            ;;
        --mode)
            mapfile -t COMPREPLY < <(compgen -W "static_anisotropic static_isotropic laminar_flow two_component" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# xpcsjax-post-install completion
_xpcsjax_post_install() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--interactive --shell --no-completion --no-xla --xla-mode --verbose --help"

    case "$prev" in
        --shell|-s)
            mapfile -t COMPREPLY < <(compgen -W "bash zsh fish" -- "${cur}")
            return
            ;;
        --xla-mode)
            mapfile -t COMPREPLY < <(compgen -W "auto nlsq" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# xpcsjax-cleanup completion
_xpcsjax_cleanup() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--dry-run --force --interactive --verbose --help"

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# xpcsjax-validate completion
_xpcsjax_validate() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--verbose --json --help"

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# xpcsjax-config-xla completion
_xpcsjax_config_xla() {
    # shellcheck disable=SC2034  # words/cword used by _init_completion
    local cur prev words cword
    _init_completion -s || return

    local opts="--threads --no-x64 --debug --info --help"

    case "$prev" in
        --threads)
            local cpu_count
            cpu_count=$(nproc 2>/dev/null || echo 4)
            mapfile -t COMPREPLY < <(compgen -W "1 2 4 8 ${cpu_count}" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# Register completions for full names
complete -F _xpcsjax xpcsjax
complete -F _xpcsjax_config xpcsjax-config
complete -F _xpcsjax_config_xla xpcsjax-config-xla
complete -F _xpcsjax_post_install xpcsjax-post-install
complete -F _xpcsjax_cleanup xpcsjax-cleanup
complete -F _xpcsjax_validate xpcsjax-validate

# Short aliases (xj = xpcsjax)
complete -F _xpcsjax xj
complete -F _xpcsjax_config xj-config
complete -F _xpcsjax_config_xla xj-config-xla
complete -F _xpcsjax_post_install xj-post-install
complete -F _xpcsjax_cleanup xj-cleanup
complete -F _xpcsjax_validate xj-validate

# Plot-only shortcuts
complete -F _xpcsjax xjexp
complete -F _xpcsjax xjsim
