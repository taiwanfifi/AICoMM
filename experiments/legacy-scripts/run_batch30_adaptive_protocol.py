#!/usr/bin/env python3
"""
Batch 30: Adaptive Protocol Simulation

Simulates an adaptive semantic transport protocol that chooses the optimal
compression level based on current bandwidth conditions.

Uses quality-bandwidth curves from Paper A experiments (empirical data)
to build the protocol's decision function.

Experiments:
  1. Quality-bandwidth Pareto frontier construction
  2. Time-varying bandwidth simulation (Markov chain)
  3. Policy comparison: static vs adaptive vs progressive
  4. Multi-agent resource allocation
  5. Scout model protocol integration

No GPU needed — pure simulation using Paper A's empirical results.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
# Paper A Empirical Data (from batch 25, normalized F1)
# =========================================================================

# Quality (% of baseline F1) at different compression levels
# From Tables 2, 6 of Paper A
QUALITY_BANDWIDTH = {
    'Qwen-7B': {
        # (bandwidth_fraction, quality_pct, method)
        'configs': [
            {'name': 'BF16',       'bw_frac': 1.000, 'quality': 100.0, 'method': 'none'},
            {'name': 'INT8',       'bw_frac': 0.500, 'quality': 99.6,  'method': 'int8'},
            {'name': 'Mixed-INT4', 'bw_frac': 0.277, 'quality': 107.1, 'method': 'mixed_int4'},
            {'name': 'INT4',       'bw_frac': 0.250, 'quality': 96.2,  'method': 'int4'},
            {'name': 'Q2C-75+BF16','bw_frac': 0.750, 'quality': 70.1,  'method': 'q2c_75_fp16'},
            {'name': 'Q2C-75+INT8','bw_frac': 0.375, 'quality': 72.7,  'method': 'q2c_75_int8'},
            {'name': 'Q2C-50+BF16','bw_frac': 0.500, 'quality': 41.8,  'method': 'q2c_50_fp16'},
            {'name': 'Q2C-50+INT8','bw_frac': 0.250, 'quality': 45.2,  'method': 'q2c_50_int8'},
            {'name': 'Q2C-25+BF16','bw_frac': 0.250, 'quality': 39.7,  'method': 'q2c_25_fp16'},
        ],
        'kv_size_bf16_mb': 9.3,
        'prefill_ms': 18,
        'quant_ms': 3,
    },
    'Mistral-7B': {
        'configs': [
            {'name': 'BF16',       'bw_frac': 1.000, 'quality': 100.0, 'method': 'none'},
            {'name': 'INT8',       'bw_frac': 0.500, 'quality': 99.8,  'method': 'int8'},
            {'name': 'Mixed-INT4', 'bw_frac': 0.277, 'quality': 93.5,  'method': 'mixed_int4'},
            {'name': 'INT4',       'bw_frac': 0.250, 'quality': 96.1,  'method': 'int4'},
            {'name': 'Q2C-75+BF16','bw_frac': 0.750, 'quality': 100.5, 'method': 'q2c_75_fp16'},
            {'name': 'Q2C-50+BF16','bw_frac': 0.500, 'quality': 88.9,  'method': 'q2c_50_fp16'},
            {'name': 'Q2C-25+BF16','bw_frac': 0.250, 'quality': 57.8,  'method': 'q2c_25_fp16'},
        ],
        'kv_size_bf16_mb': 22.4,
        'prefill_ms': 22,
        'quant_ms': 3.3,
    },
    'Qwen-14B': {
        'configs': [
            {'name': 'BF16',       'bw_frac': 1.000, 'quality': 100.0, 'method': 'none'},
            {'name': 'INT8',       'bw_frac': 0.500, 'quality': 100.0, 'method': 'int8'},
            {'name': 'Mixed-INT4', 'bw_frac': 0.277, 'quality': 95.6,  'method': 'mixed_int4'},
            {'name': 'INT4',       'bw_frac': 0.250, 'quality': 95.5,  'method': 'int4'},
            {'name': 'Q2C-75+BF16','bw_frac': 0.750, 'quality': 90.7,  'method': 'q2c_75_fp16'},
            {'name': 'Q2C-50+BF16','bw_frac': 0.500, 'quality': 78.0,  'method': 'q2c_50_fp16'},
            {'name': 'Q2C-25+BF16','bw_frac': 0.250, 'quality': 63.9,  'method': 'q2c_25_fp16'},
        ],
        'kv_size_bf16_mb': 31.9,
        'prefill_ms': 57,
        'quant_ms': 4.8,
    },
}

# Scout model data (from batch 7c v2, to be updated with batch 28)
SCOUT_DATA = {
    '3B_to_7B': {
        'overlap_75': 91.5,  # position overlap at 75% retention
        'overlap_50': 86.3,  # at 50%
        'f1_loss_75': -0.008,
        'f1_loss_50': -0.046,
        'index_size_bytes_per_token': 4,  # int32
    }
}


# =========================================================================
# 1. Pareto Frontier Construction
# =========================================================================
def build_pareto_frontier(model_name):
    """Build quality-bandwidth Pareto frontier from empirical data."""
    configs = QUALITY_BANDWIDTH[model_name]['configs']

    # Sort by bandwidth (ascending)
    sorted_configs = sorted(configs, key=lambda c: c['bw_frac'])

    # Find Pareto-optimal points (no config with less BW gives higher quality)
    pareto = []
    best_quality = -1
    for c in sorted(configs, key=lambda c: c['bw_frac']):
        if c['quality'] > best_quality:
            pareto.append(c)
            best_quality = c['quality']

    return pareto


# =========================================================================
# 2. Markov Chain Bandwidth Simulator
# =========================================================================
class BandwidthSimulator:
    """Simulate time-varying bandwidth using a Markov chain."""

    # States: bandwidth in Mbps
    STATES = [5, 10, 25, 50, 100, 200]

    # Transition matrix (designed for realistic wireless conditions)
    # Higher probability to stay in same state or move to adjacent
    TRANSITION = np.array([
        [0.50, 0.30, 0.10, 0.05, 0.03, 0.02],  # 5 Mbps
        [0.20, 0.40, 0.25, 0.10, 0.03, 0.02],  # 10 Mbps
        [0.05, 0.15, 0.45, 0.25, 0.07, 0.03],  # 25 Mbps
        [0.03, 0.05, 0.15, 0.45, 0.25, 0.07],  # 50 Mbps
        [0.02, 0.03, 0.05, 0.15, 0.45, 0.30],  # 100 Mbps
        [0.02, 0.03, 0.05, 0.10, 0.30, 0.50],  # 200 Mbps
    ])

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.current_state = 2  # Start at 25 Mbps
        assert np.allclose(self.TRANSITION.sum(axis=1), 1.0)

    def step(self):
        """Transition to next bandwidth state."""
        probs = self.TRANSITION[self.current_state]
        self.current_state = self.rng.choice(len(self.STATES), p=probs)
        return self.STATES[self.current_state]

    def generate_trace(self, num_steps):
        """Generate a bandwidth trace."""
        trace = []
        for _ in range(num_steps):
            bw = self.step()
            trace.append(bw)
        return trace


# =========================================================================
# 3. Protocol Policies
# =========================================================================
def select_compression(bandwidth_mbps, model_name, deadline_ms, policy='adaptive'):
    """
    Select compression configuration given current bandwidth and deadline.

    Args:
        bandwidth_mbps: Current bandwidth in Mbps
        model_name: Model name
        deadline_ms: Deadline in ms
        policy: 'static_int8', 'static_int4', 'adaptive', 'progressive', 'scout'

    Returns:
        (config_name, quality_pct, tx_time_ms, total_time_ms, meets_deadline)
    """
    model = QUALITY_BANDWIDTH[model_name]
    kv_size_mb = model['kv_size_bf16_mb']
    prefill_ms = model['prefill_ms']
    quant_ms = model['quant_ms']
    decode_ms = 167  # Average decode time from Paper A

    if policy == 'static_int8':
        bw_frac = 0.5
        tx_ms = (kv_size_mb * bw_frac * 8) / bandwidth_mbps * 1000
        total = prefill_ms + quant_ms + tx_ms + decode_ms
        cfg = next(c for c in model['configs'] if c['method'] == 'int8')
        return cfg['name'], cfg['quality'], tx_ms, total, total <= deadline_ms

    elif policy == 'static_int4':
        bw_frac = 0.25
        tx_ms = (kv_size_mb * bw_frac * 8) / bandwidth_mbps * 1000
        total = prefill_ms + quant_ms + tx_ms + decode_ms
        cfg = next(c for c in model['configs'] if c['method'] == 'int4')
        return cfg['name'], cfg['quality'], tx_ms, total, total <= deadline_ms

    elif policy == 'adaptive':
        # Try configs from highest quality to lowest, pick best that meets deadline
        configs = sorted(model['configs'], key=lambda c: -c['quality'])
        for cfg in configs:
            tx_ms = (kv_size_mb * cfg['bw_frac'] * 8) / bandwidth_mbps * 1000
            overhead = prefill_ms + (quant_ms if 'int' in cfg['method'] else 0)
            total = overhead + tx_ms + decode_ms
            if total <= deadline_ms:
                return cfg['name'], cfg['quality'], tx_ms, total, True

        # Nothing meets deadline — use smallest config
        cfg = min(model['configs'], key=lambda c: c['bw_frac'])
        tx_ms = (kv_size_mb * cfg['bw_frac'] * 8) / bandwidth_mbps * 1000
        total = prefill_ms + quant_ms + tx_ms + decode_ms
        return cfg['name'], cfg['quality'], tx_ms, total, False

    elif policy == 'scout':
        # Scout model: only transmit position indices
        # Assume 50% retention, 4 bytes per index
        avg_context = 170  # tokens
        idx_bytes = avg_context * 0.5 * 4
        tx_ms = (idx_bytes * 8) / (bandwidth_mbps * 1e6) * 1000
        # Cloud needs to run its own prefill
        cloud_prefill = prefill_ms * 1.5  # Cloud model might be bigger
        total = 10 + tx_ms + cloud_prefill + decode_ms  # 10ms = edge 3B prefill
        # Quality: ~95% of cloud's own selection (from batch 7c/28 data)
        quality = 95.0  # Approximate, will be updated with batch 28 results
        return 'Scout-50%', quality, tx_ms, total, total <= deadline_ms

    elif policy == 'no_transfer':
        # Edge generates locally (no cloud help)
        # Quality = edge model quality / cloud model quality
        quality_ratio = {'Qwen-7B': 75.0, 'Mistral-7B': 60.0, 'Qwen-14B': 65.0}
        quality = quality_ratio.get(model_name, 70.0)
        return 'Local-Edge', quality, 0, prefill_ms + decode_ms, True

    raise ValueError(f"Unknown policy: {policy}")


# =========================================================================
# 4. Simulation Runner
# =========================================================================
def run_protocol_simulation(model_name, num_requests=1000, deadline_ms=3000, seed=42):
    """
    Simulate protocol behavior under varying bandwidth.

    Returns per-request results for each policy.
    """
    bw_sim = BandwidthSimulator(seed=seed)
    bw_trace = bw_sim.generate_trace(num_requests)

    policies = ['static_int8', 'static_int4', 'adaptive', 'scout', 'no_transfer']
    results = {p: [] for p in policies}

    for i, bw in enumerate(bw_trace):
        for policy in policies:
            name, quality, tx_ms, total_ms, meets = select_compression(
                bw, model_name, deadline_ms, policy
            )
            results[policy].append({
                'request_idx': i,
                'bandwidth_mbps': bw,
                'config': name,
                'quality_pct': quality,
                'tx_ms': tx_ms,
                'total_ms': total_ms,
                'meets_deadline': meets,
            })

    return results, bw_trace


# =========================================================================
# 5. Multi-Agent Resource Allocation
# =========================================================================
def run_multi_agent_simulation(num_agents=4, total_bandwidth_mbps=100,
                                deadline_ms=5000, num_rounds=500, seed=42):
    """
    Simulate N agents sharing a base station's uplink bandwidth.

    Policies:
      - Equal: Each agent gets total_bw / N
      - Model-aware: Allocate proportional to KV-cache size
      - Quality-max: Optimize for max total quality
    """
    rng = np.random.RandomState(seed)

    # Assign models to agents
    model_pool = ['Qwen-7B', 'Mistral-7B', 'Qwen-14B']
    agent_models = [model_pool[i % len(model_pool)] for i in range(num_agents)]

    policies = ['equal', 'model_aware', 'quality_max']
    results = {p: [] for p in policies}

    for round_idx in range(num_rounds):
        for policy in policies:
            round_result = []

            if policy == 'equal':
                bw_per_agent = total_bandwidth_mbps / num_agents
                for agent_idx, model in enumerate(agent_models):
                    _, quality, tx, total, meets = select_compression(
                        bw_per_agent, model, deadline_ms, 'adaptive'
                    )
                    round_result.append({
                        'agent': agent_idx, 'model': model,
                        'bw_mbps': bw_per_agent, 'quality': quality,
                        'meets_deadline': meets
                    })

            elif policy == 'model_aware':
                # Allocate proportional to KV-cache size
                sizes = [QUALITY_BANDWIDTH[m]['kv_size_bf16_mb'] for m in agent_models]
                total_size = sum(sizes)
                for agent_idx, model in enumerate(agent_models):
                    bw = total_bandwidth_mbps * (sizes[agent_idx] / total_size)
                    _, quality, tx, total, meets = select_compression(
                        bw, model, deadline_ms, 'adaptive'
                    )
                    round_result.append({
                        'agent': agent_idx, 'model': model,
                        'bw_mbps': bw, 'quality': quality,
                        'meets_deadline': meets
                    })

            elif policy == 'quality_max':
                # Try different allocations, pick best total quality
                # Simple grid search over proportional allocations
                best_quality_sum = -1
                best_alloc = None

                # Generate candidate allocations
                for _ in range(100):
                    weights = rng.dirichlet(np.ones(num_agents))
                    bw_alloc = weights * total_bandwidth_mbps

                    total_q = 0
                    candidate = []
                    for agent_idx, model in enumerate(agent_models):
                        _, quality, tx, total, meets = select_compression(
                            bw_alloc[agent_idx], model, deadline_ms, 'adaptive'
                        )
                        if meets:
                            total_q += quality
                        candidate.append({
                            'agent': agent_idx, 'model': model,
                            'bw_mbps': bw_alloc[agent_idx], 'quality': quality,
                            'meets_deadline': meets
                        })

                    if total_q > best_quality_sum:
                        best_quality_sum = total_q
                        best_alloc = candidate

                round_result = best_alloc if best_alloc else round_result

            results[policy].append({
                'round': round_idx,
                'agents': round_result,
                'total_quality': sum(a['quality'] for a in round_result),
                'all_meet_deadline': all(a['meets_deadline'] for a in round_result),
            })

    return results, agent_models


# =========================================================================
# 6. Analysis and Reporting
# =========================================================================
def analyze_protocol_results(results, bw_trace, model_name, deadline_ms):
    """Analyze protocol simulation results."""
    analysis = {}

    for policy, policy_results in results.items():
        qualities = [r['quality_pct'] for r in policy_results]
        meets = [r['meets_deadline'] for r in policy_results]
        latencies = [r['total_ms'] for r in policy_results]

        analysis[policy] = {
            'avg_quality': float(np.mean(qualities)),
            'std_quality': float(np.std(qualities)),
            'min_quality': float(np.min(qualities)),
            'max_quality': float(np.max(qualities)),
            'deadline_success_rate': float(np.mean(meets)),
            'avg_latency_ms': float(np.mean(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
        }

    return analysis


def analyze_multi_agent_results(results, agent_models):
    """Analyze multi-agent allocation results."""
    analysis = {}

    for policy, rounds in results.items():
        total_qs = [r['total_quality'] for r in rounds]
        all_meets = [r['all_meet_deadline'] for r in rounds]

        # Per-agent quality
        per_agent_q = {i: [] for i in range(len(agent_models))}
        for r in rounds:
            for a in r['agents']:
                per_agent_q[a['agent']].append(a['quality'])

        analysis[policy] = {
            'avg_total_quality': float(np.mean(total_qs)),
            'avg_all_meet_deadline': float(np.mean(all_meets)),
            'per_agent_avg_quality': {
                i: float(np.mean(qs)) for i, qs in per_agent_q.items()
            },
            'jain_fairness': float(
                np.mean(total_qs)**2 / (len(agent_models) * np.mean([q**2 for q in total_qs]))
            ) if np.mean(total_qs) > 0 else 0,
        }

    return analysis


# =========================================================================
# Main
# =========================================================================
def main():
    logger.info("="*70)
    logger.info("Batch 30: Adaptive Protocol Simulation")
    logger.info("="*70)

    all_results = {}

    # ---- Experiment 1: Per-model protocol simulation ----
    for model_name in ['Qwen-7B', 'Mistral-7B', 'Qwen-14B']:
        for deadline_ms in [1000, 2000, 3000, 5000]:
            logger.info(f"\n--- {model_name}, deadline={deadline_ms}ms ---")
            results, bw_trace = run_protocol_simulation(
                model_name, num_requests=1000, deadline_ms=deadline_ms
            )
            analysis = analyze_protocol_results(results, bw_trace, model_name, deadline_ms)

            key = f"{model_name}_deadline{deadline_ms}"
            all_results[key] = analysis

            for policy, stats in analysis.items():
                logger.info(f"  {policy:15s}: Q={stats['avg_quality']:5.1f}% "
                          f"deadline_ok={stats['deadline_success_rate']*100:5.1f}% "
                          f"lat_p50={stats['p50_latency_ms']:7.0f}ms "
                          f"lat_p95={stats['p95_latency_ms']:7.0f}ms")

    # ---- Experiment 2: Pareto frontiers ----
    pareto_results = {}
    for model_name in ['Qwen-7B', 'Mistral-7B', 'Qwen-14B']:
        pareto = build_pareto_frontier(model_name)
        pareto_results[model_name] = pareto
        logger.info(f"\nPareto frontier for {model_name}:")
        for p in pareto:
            logger.info(f"  {p['name']:20s}: BW={p['bw_frac']*100:5.1f}% Q={p['quality']:5.1f}%")

    # ---- Experiment 3: Multi-agent allocation ----
    for n_agents in [2, 4, 8]:
        for total_bw in [50, 100, 200]:
            logger.info(f"\n--- {n_agents} agents, {total_bw} Mbps total ---")
            ma_results, agent_models = run_multi_agent_simulation(
                num_agents=n_agents, total_bandwidth_mbps=total_bw,
                deadline_ms=5000, num_rounds=500
            )
            ma_analysis = analyze_multi_agent_results(ma_results, agent_models)

            key = f"multiagent_{n_agents}agents_{total_bw}mbps"
            all_results[key] = ma_analysis

            for policy, stats in ma_analysis.items():
                logger.info(f"  {policy:15s}: total_Q={stats['avg_total_quality']:6.1f} "
                          f"all_ok={stats['avg_all_meet_deadline']*100:5.1f}%")

    # ---- Experiment 4: Bandwidth trace analysis ----
    bw_sim = BandwidthSimulator(seed=42)
    trace = bw_sim.generate_trace(10000)
    bw_stats = {
        'mean': float(np.mean(trace)),
        'std': float(np.std(trace)),
        'median': float(np.median(trace)),
        'p5': float(np.percentile(trace, 5)),
        'p95': float(np.percentile(trace, 95)),
        'distribution': {str(s): int(trace.count(s) if isinstance(trace, list) else
                                     np.sum(np.array(trace) == s))
                        for s in BandwidthSimulator.STATES}
    }
    all_results['bandwidth_trace_stats'] = bw_stats
    logger.info(f"\nBandwidth trace stats: mean={bw_stats['mean']:.1f} "
               f"median={bw_stats['median']:.1f} "
               f"p5={bw_stats['p5']:.1f} p95={bw_stats['p95']:.1f}")

    # Save all results
    output = {
        'metadata': {
            'experiment': 'batch30_adaptive_protocol',
            'timestamp': datetime.now().isoformat(),
            'description': 'Adaptive protocol simulation using Paper A empirical data',
        },
        'protocol_simulation': {k: v for k, v in all_results.items()
                                if not k.startswith('multiagent')
                                and k != 'bandwidth_trace_stats'},
        'multi_agent': {k: v for k, v in all_results.items()
                        if k.startswith('multiagent')},
        'pareto_frontiers': {k: v for k, v in pareto_results.items()},
        'bandwidth_stats': bw_stats,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f'batch30_adaptive_protocol_{ts}.json'
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\n[SAVED] → {result_path}")

    return output


if __name__ == '__main__':
    main()
