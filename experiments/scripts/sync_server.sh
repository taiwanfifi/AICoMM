#!/bin/bash
# ============================================================================
# Sync scripts to/from GPU servers
#
# Usage:
#   ./sync_server.sh push [a6000|blackwell|all]
#   ./sync_server.sh pull [a6000|blackwell|all]
#   ./sync_server.sh status [a6000|blackwell|all]
#
# Default: operates on all servers
# ============================================================================

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

# Server configurations
A6000_HOST="38.29.145.24"
A6000_PORT="40434"
A6000_USER="root"
A6000_DIR="/workspace/experiments"

BLACKWELL_HOST="104.6.154.179"
BLACKWELL_PORT="60539"
BLACKWELL_USER="root"
BLACKWELL_DIR="/workspace/experiments"

do_push() {
    local name=$1 host=$2 port=$3 user=$4 dir=$5
    echo "=== Pushing to $name ($host:$port) ==="
    ssh -p "$port" $SSH_OPTS "$user@$host" "mkdir -p $dir/scripts $dir/results $dir/logs" 2>/dev/null
    scp -P "$port" $SSH_OPTS "$LOCAL_DIR/scripts/exp_utils.py" "$LOCAL_DIR/scripts/run_exp_"*.py \
        "$user@$host:$dir/scripts/" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  OK: $(ssh -p "$port" $SSH_OPTS "$user@$host" "ls $dir/scripts/run_exp_*.py 2>/dev/null | wc -l") scripts synced"
    else
        echo "  FAILED: Could not connect to $name"
    fi
}

do_pull() {
    local name=$1 host=$2 port=$3 user=$4 dir=$5
    echo "=== Pulling from $name ($host:$port) ==="
    mkdir -p "$LOCAL_DIR/results" "$LOCAL_DIR/logs"

    # Pull JSON results
    scp -P "$port" $SSH_OPTS "$user@$host:$dir/results/exp_*.json" "$LOCAL_DIR/results/" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  Results: $(ls "$LOCAL_DIR/results"/exp_*.json 2>/dev/null | wc -l) files"
    else
        echo "  No results (or connection failed)"
    fi

    # Pull logs
    scp -P "$port" $SSH_OPTS "$user@$host:$dir/logs/*.log" "$LOCAL_DIR/logs/" 2>/dev/null
}

do_status() {
    local name=$1 host=$2 port=$3 user=$4 dir=$5
    echo "=== $name Status ($host:$port) ==="
    ssh -p "$port" $SSH_OPTS "$user@$host" "
        echo '--- GPU ---'
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU'
        echo '--- Screen sessions ---'
        screen -ls 2>/dev/null || echo 'No screen sessions'
        echo '--- Running experiments ---'
        ps aux | grep 'python3.*run_exp' | grep -v grep || echo 'None running'
        echo '--- Results ---'
        ls -lt $dir/results/exp_*.json 2>/dev/null | head -5 || echo 'No results'
        echo '--- Disk ---'
        df -h / /dev/shm 2>/dev/null | tail -2
    " 2>&1 || echo "  Could not connect to $name"
    echo ""
}

run_for_servers() {
    local action=$1 target=${2:-all}
    case "$target" in
        a6000)
            $action "A6000" "$A6000_HOST" "$A6000_PORT" "$A6000_USER" "$A6000_DIR"
            ;;
        blackwell)
            $action "Blackwell" "$BLACKWELL_HOST" "$BLACKWELL_PORT" "$BLACKWELL_USER" "$BLACKWELL_DIR"
            ;;
        all)
            $action "A6000" "$A6000_HOST" "$A6000_PORT" "$A6000_USER" "$A6000_DIR"
            $action "Blackwell" "$BLACKWELL_HOST" "$BLACKWELL_PORT" "$BLACKWELL_USER" "$BLACKWELL_DIR"
            ;;
        *)
            echo "Unknown server: $target (use a6000, blackwell, or all)"
            exit 1
            ;;
    esac
}

case "$1" in
    push)   run_for_servers do_push "$2" ;;
    pull)   run_for_servers do_pull "$2" ;;
    status) run_for_servers do_status "$2" ;;
    *)
        echo "Usage: $0 {push|pull|status} [a6000|blackwell|all]"
        echo "  push   - Push experiment scripts to server(s)"
        echo "  pull   - Pull results from server(s)"
        echo "  status - Check server status"
        exit 1
        ;;
esac
