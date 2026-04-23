#!/usr/bin/env bash

run_pipeline_steps() {
    local pipeline_label="$1"
    local script_dir="$2"
    local project_root="$3"
    local repo_root="$4"
    local log_root="$5"
    local py="$6"
    local oneg="$7"
    local scripts_array_name="$8"
    local -n run_scripts="$scripts_array_name"

    mkdir -p "$log_root"

    echo "============================================================"
    echo "$pipeline_label"
    echo "Script dir : $script_dir"
    echo "Project    : $project_root"
    echo "Repo       : $repo_root"
    echo "PY         : $py"
    echo "ONEG       : $oneg"
    echo "PYTHONPATH : ${PYTHONPATH:-}"
    echo "Log root   : $log_root"
    echo "============================================================"

    local total=0
    local ok=0
    local failed=0
    local -a failed_steps=()

    local script_name script_path log_path status item
    for script_name in "${run_scripts[@]}"; do
        total=$((total + 1))
        script_path="$script_dir/$script_name"
        log_path="$log_root/${script_name%.sh}.log"

        echo
        echo "============================================================"
        echo "Step $total"
        echo "Script : $script_name"
        echo "Log    : $log_path"
        echo "============================================================"

        if [[ ! -f "$script_path" ]]; then
            failed=$((failed + 1))
            failed_steps+=("missing :: $script_name")
            echo "ERROR: script not found: $script_path" >&2
            continue
        fi

        if PY="$py" LOG_ROOT="$log_root" ONEG="$oneg" bash "$script_path" >"$log_path" 2>&1; then
            ok=$((ok + 1))
            echo "OK: $script_name"
        else
            status=$?
            failed=$((failed + 1))
            failed_steps+=("exit $status :: $script_name")
            echo "WARNING: $script_name failed with exit code $status" >&2
            echo "Check log: $log_path" >&2
        fi
    done

    echo
    echo "============================================================"
    echo "Finished."
    echo "  Total steps : $total"
    echo "  Successful  : $ok"
    echo "  Failed      : $failed"

    if (( failed > 0 )); then
        echo
        echo "Failed steps:"
        for item in "${failed_steps[@]}"; do
            echo "  - $item"
        done
    fi
}
