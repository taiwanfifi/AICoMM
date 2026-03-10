#!/bin/bash
# ============================================================================
# Run experiment in a screen session (resilient to SSH disconnects)
#
# Usage:
#   bash run_experiment.sh run_exp_q2c_ablation.py
#   bash run_experiment.sh run_exp_perplexity.py
#   bash run_experiment.sh all   # Run all in sequence
#
# Experiments run inside screen, so they survive SSH disconnects.
# Re-attach with: screen -r exp_<name>
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Source environment
source "$SCRIPT_DIR/../.env" 2>/dev/null || true
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

# Ensure results dir is accessible from working directory
cd "$SCRIPT_DIR"
ln -sf "$RESULTS_DIR" results 2>/dev/null || true

run_single() {
    local script="$1"
    local name="${script%.py}"
    local session="exp_${name}"
    local logfile="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "Starting $script in screen session '$session'"
    echo "Log: $logfile"

    screen -dmS "$session" bash -c "
        cd $SCRIPT_DIR
        source $SCRIPT_DIR/../.env 2>/dev/null || true
        export TRANSFORMERS_NO_TF=1
        export TOKENIZERS_PARALLELISM=false
        python3 $script 2>&1 | tee $logfile
        echo ''
        echo '=== EXPERIMENT COMPLETE ==='
        echo \"Finished at: \$(date)\"
    "

    echo "Screen session started. Reattach with: screen -r $session"
}

case "$1" in
    all)
        echo "=== Running all experiments in sequence ==="
        # Phase 1
        run_single "run_exp_q2c_ablation.py"
        echo "Waiting for Q2C ablation to finish..."
        while screen -ls | grep -q "exp_run_exp_q2c_ablation"; do sleep 60; done

        run_single "run_exp_perplexity.py"
        echo "Waiting for perplexity to finish..."
        while screen -ls | grep -q "exp_run_exp_perplexity"; do sleep 60; done

        # Phase 2
        run_single "run_exp_scout_n200.py"
        while screen -ls | grep -q "exp_run_exp_scout_n200"; do sleep 60; done

        run_single "run_exp_scout_long_context.py"
        while screen -ls | grep -q "exp_run_exp_scout_long_context"; do sleep 60; done

        run_single "run_exp_scout_cross_family.py"
        while screen -ls | grep -q "exp_run_exp_scout_cross_family"; do sleep 60; done

        run_single "run_exp_scout_multitask.py"
        while screen -ls | grep -q "exp_run_exp_scout_multitask"; do sleep 60; done

        # Phase 3
        run_single "run_exp_attention_entropy.py"
        while screen -ls | grep -q "exp_run_exp_attention_entropy"; do sleep 60; done

        # Phase 4
        run_single "run_exp_protocol_real_traces.py"
        while screen -ls | grep -q "exp_run_exp_protocol_real_traces"; do sleep 60; done

        run_single "run_exp_hybrid_mode.py"

        echo "All experiments queued."
        ;;

    *.py)
        run_single "$1"
        ;;

    *)
        echo "Usage: $0 {script_name.py | all}"
        echo ""
        echo "Available experiments:"
        ls run_exp_*.py 2>/dev/null | sed 's/^/  /'
        exit 1
        ;;
esac
