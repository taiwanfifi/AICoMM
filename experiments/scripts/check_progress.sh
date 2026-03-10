#!/bin/bash
# Quick progress check for Round 1 experiments
# Usage: bash check_progress.sh

SSH_CMD="ssh -o StrictHostKeyChecking=no -p 10948 root@192.165.134.28"

echo "=== Experiment Progress $(date) ==="

$SSH_CMD "
echo '--- Process Status ---'
ps aux | grep python3 | grep -v grep | awk '{print \$NF, \"PID:\"\$2, \"CPU:\"\$3\"%\", \"MEM:\"\$4\"%\"}'
echo ''

echo '--- GPU Status ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ''

echo '--- Log Summary ---'
if [ -f /workspace/round1.log ]; then
    echo 'Last 5 lines:'
    tail -5 /workspace/round1.log
    echo ''
    echo 'Progress markers:'
    grep -E '(Edge \[|Cloud \[|RESULTS|SAVED|E[123]\]|Done|Total time)' /workspace/round1.log | tail -10
    echo ''
    echo 'Errors:'
    grep -c 'ERROR\|Traceback' /workspace/round1.log
else
    echo 'No log file found'
fi
echo ''

echo '--- Result Files ---'
ls -la /workspace/AI-Comm/experiments/results/exp_*.json 2>/dev/null | awk '{print \$NF, \$5}'
"
