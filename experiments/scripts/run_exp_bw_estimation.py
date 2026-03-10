#!/usr/bin/env python3
"""
Experiment: Bandwidth Estimation Validation

Validates the EWMA bandwidth estimator used in the Scout adaptive protocol.
Tests:
  1. Estimation error distribution across 5G scenarios
  2. Mode selection accuracy vs. oracle (perfect BW knowledge)
  3. Alpha sensitivity (0.1, 0.2, 0.3, 0.5, 0.7)
  4. Conservative factor impact (0.7, 0.8, 0.9, 1.0)
  5. Quality and deadline impact of estimation errors

No GPU needed — pure simulation using Paper A empirical data.
"""

import sys
import json
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# =========================================================================
# Reuse trace generators from protocol experiment
# =========================================================================

class SyntheticLumos5GTrace:
    """Synthetic 5G bandwidth traces based on Lumos5G statistics."""

    def __init__(self, scenario='urban_mmwave', seed=42):
        self.rng = np.random.RandomState(seed)
        self.scenario = scenario
        self.params = {
            'urban_mmwave': {
                'mean_mbps': 200, 'std_mbps': 150,
                'min_mbps': 2, 'max_mbps': 800,
                'autocorrelation': 0.85, 'outage_prob': 0.05, 'outage_bw': 3,
            },
            'urban_sub6': {
                'mean_mbps': 60, 'std_mbps': 30,
                'min_mbps': 5, 'max_mbps': 150,
                'autocorrelation': 0.90, 'outage_prob': 0.02, 'outage_bw': 5,
            },
            'suburban_mixed': {
                'mean_mbps': 80, 'std_mbps': 50,
                'min_mbps': 3, 'max_mbps': 300,
                'autocorrelation': 0.88, 'outage_prob': 0.03, 'outage_bw': 4,
            },
            'vehicular': {
                'mean_mbps': 50, 'std_mbps': 60,
                'min_mbps': 1, 'max_mbps': 400,
                'autocorrelation': 0.70, 'outage_prob': 0.10, 'outage_bw': 2,
            },
        }[scenario]
        self.current_bw = self.params['mean_mbps']

    def step(self):
        p = self.params
        if self.rng.random() < p['outage_prob']:
            self.current_bw = p['outage_bw']
            return self.current_bw
        innovation = self.rng.normal(0, p['std_mbps'] * math.sqrt(1 - p['autocorrelation']**2))
        self.current_bw = (p['autocorrelation'] * self.current_bw +
                          (1 - p['autocorrelation']) * p['mean_mbps'] + innovation)
        self.current_bw = max(p['min_mbps'], min(p['max_mbps'], self.current_bw))
        return self.current_bw

    def generate_trace(self, num_steps):
        trace = []
        for _ in range(num_steps):
            trace.append(self.step())
        return np.array(trace)


# =========================================================================
# Operating modes (from JSAC paper Table 2)
# =========================================================================

# Qwen-7B operating modes (primary evaluation model)
OPERATING_MODES = [
    {'name': 'Full BF16',  'payload_mb': 9.7,    'quality': 100.0, 'quant_ms': 0},
    {'name': 'INT8',       'payload_mb': 4.7,     'quality': 99.4,  'quant_ms': 3},
    {'name': 'Mixed-INT4', 'payload_mb': 2.6,     'quality': 93.6,  'quant_ms': 3},
    {'name': 'INT4',       'payload_mb': 2.3,     'quality': 68.7,  'quant_ms': 3},
    {'name': 'Scout',      'payload_mb': 336e-6,  'quality': 90.0,  'quant_ms': 0},
]

PREFILL_MS = 18      # Edge prefill (Qwen-7B)
CLOUD_PREFILL_MS = 57  # Cloud re-prefill (Qwen-14B)
DECODE_MS = 167       # Average decode time


def total_latency(mode, bw_mbps):
    """Compute total end-to-end latency for a mode at given bandwidth."""
    tx_ms = (mode['payload_mb'] * 8) / bw_mbps * 1000 if bw_mbps > 0 else float('inf')
    if mode['name'] == 'Scout':
        return PREFILL_MS + tx_ms + CLOUD_PREFILL_MS + DECODE_MS
    else:
        return PREFILL_MS + mode['quant_ms'] + tx_ms + DECODE_MS


def select_mode_oracle(true_bw, deadline_ms):
    """Oracle mode selection with perfect bandwidth knowledge."""
    for mode in OPERATING_MODES:  # Already sorted by decreasing quality
        if total_latency(mode, true_bw) <= deadline_ms:
            return mode
    return OPERATING_MODES[-1]  # Scout as fallback


def select_mode_estimated(estimated_bw, deadline_ms):
    """Mode selection using estimated bandwidth."""
    for mode in OPERATING_MODES:
        if total_latency(mode, estimated_bw) <= deadline_ms:
            return mode
    return OPERATING_MODES[-1]


# =========================================================================
# EWMA Bandwidth Estimator
# =========================================================================

class EWMAEstimator:
    """Exponentially Weighted Moving Average bandwidth estimator."""

    def __init__(self, alpha=0.3, conservative_factor=0.8):
        self.alpha = alpha
        self.cf = conservative_factor
        self.estimate = None

    def update(self, measured_bw):
        if self.estimate is None:
            self.estimate = measured_bw
        else:
            self.estimate = self.alpha * measured_bw + (1 - self.alpha) * self.estimate
        return self.estimate * self.cf

    def reset(self):
        self.estimate = None


# =========================================================================
# Experiment 1: Estimation Error Distribution
# =========================================================================

def exp_estimation_error(num_steps=5000, alpha=0.3, cf=0.8):
    """Compute estimation error across all scenarios."""
    scenarios = ['urban_mmwave', 'urban_sub6', 'suburban_mixed', 'vehicular']
    results = {}

    for scenario in scenarios:
        trace_gen = SyntheticLumos5GTrace(scenario, seed=42)
        trace = trace_gen.generate_trace(num_steps)

        estimator = EWMAEstimator(alpha=alpha, conservative_factor=cf)
        errors = []
        relative_errors = []
        estimates = []

        for i, true_bw in enumerate(trace):
            est_bw = estimator.update(true_bw)
            estimates.append(est_bw)
            # Error = estimated - true (negative = underestimate = conservative)
            err = est_bw - true_bw
            errors.append(err)
            if true_bw > 0:
                relative_errors.append(err / true_bw)

        errors = np.array(errors)
        relative_errors = np.array(relative_errors)

        results[scenario] = {
            'trace_stats': {
                'mean': float(np.mean(trace)),
                'std': float(np.std(trace)),
                'median': float(np.median(trace)),
                'p5': float(np.percentile(trace, 5)),
                'p95': float(np.percentile(trace, 95)),
            },
            'absolute_error': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'median': float(np.median(errors)),
                'p5': float(np.percentile(errors, 5)),
                'p95': float(np.percentile(errors, 95)),
                'mae': float(np.mean(np.abs(errors))),
            },
            'relative_error': {
                'mean': float(np.mean(relative_errors)),
                'std': float(np.std(relative_errors)),
                'median': float(np.median(relative_errors)),
                'p5': float(np.percentile(relative_errors, 5)),
                'p95': float(np.percentile(relative_errors, 95)),
                'mape': float(np.mean(np.abs(relative_errors))),
            },
            'underestimate_rate': float(np.mean(np.array(errors) < 0)),
        }

    return results


# =========================================================================
# Experiment 2: Mode Selection Accuracy vs Oracle
# =========================================================================

def exp_mode_selection_accuracy(num_steps=5000, alpha=0.3, cf=0.8):
    """Compare mode selection with estimation vs. oracle."""
    scenarios = ['urban_mmwave', 'urban_sub6', 'suburban_mixed', 'vehicular']
    deadlines = [500, 1000, 2000, 3000, 5000]
    results = {}

    for scenario in scenarios:
        trace_gen = SyntheticLumos5GTrace(scenario, seed=42)
        trace = trace_gen.generate_trace(num_steps)
        scenario_results = {}

        for deadline in deadlines:
            estimator = EWMAEstimator(alpha=alpha, conservative_factor=cf)

            exact_match = 0
            quality_match = 0     # Same quality even if different mode
            overconservative = 0  # Chose lower mode than needed
            underconservative = 0 # Chose higher mode, missed deadline
            oracle_quality_sum = 0
            est_quality_sum = 0
            oracle_deadline_met = 0
            est_deadline_met = 0

            for true_bw in trace:
                est_bw = estimator.update(true_bw)

                oracle_mode = select_mode_oracle(true_bw, deadline)
                est_mode = select_mode_estimated(est_bw, deadline)

                # Check if oracle choice meets deadline (with true BW)
                oracle_lat = total_latency(oracle_mode, true_bw)
                # Check if estimated choice actually meets deadline (with TRUE BW)
                est_actual_lat = total_latency(est_mode, true_bw)

                oracle_quality_sum += oracle_mode['quality']
                est_quality_sum += est_mode['quality']

                if oracle_lat <= deadline:
                    oracle_deadline_met += 1
                if est_actual_lat <= deadline:
                    est_deadline_met += 1

                if oracle_mode['name'] == est_mode['name']:
                    exact_match += 1
                elif oracle_mode['quality'] == est_mode['quality']:
                    quality_match += 1
                elif est_mode['quality'] < oracle_mode['quality']:
                    overconservative += 1
                else:
                    underconservative += 1

            n = len(trace)
            scenario_results[f'd{deadline}ms'] = {
                'exact_match_rate': round(exact_match / n, 4),
                'quality_match_rate': round((exact_match + quality_match) / n, 4),
                'overconservative_rate': round(overconservative / n, 4),
                'underconservative_rate': round(underconservative / n, 4),
                'oracle_avg_quality': round(oracle_quality_sum / n, 2),
                'estimated_avg_quality': round(est_quality_sum / n, 2),
                'quality_gap': round((oracle_quality_sum - est_quality_sum) / n, 2),
                'oracle_deadline_rate': round(oracle_deadline_met / n, 4),
                'estimated_deadline_rate': round(est_deadline_met / n, 4),
            }

        results[scenario] = scenario_results

    return results


# =========================================================================
# Experiment 3: Alpha Sensitivity
# =========================================================================

def exp_alpha_sensitivity(num_steps=5000, cf=0.8):
    """Test sensitivity to EWMA alpha parameter."""
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    scenario = 'vehicular'  # Most challenging: high variability, low autocorrelation
    deadline = 2000  # Medium deadline

    results = {}
    for alpha in alphas:
        trace_gen = SyntheticLumos5GTrace(scenario, seed=42)
        trace = trace_gen.generate_trace(num_steps)
        estimator = EWMAEstimator(alpha=alpha, conservative_factor=cf)

        errors = []
        mode_matches = 0
        quality_sum_oracle = 0
        quality_sum_est = 0
        deadline_met_oracle = 0
        deadline_met_est = 0
        mode_switches = 0
        prev_mode = None

        for true_bw in trace:
            est_bw = estimator.update(true_bw)
            errors.append(est_bw - true_bw)

            oracle_mode = select_mode_oracle(true_bw, deadline)
            est_mode = select_mode_estimated(est_bw, deadline)

            if oracle_mode['name'] == est_mode['name']:
                mode_matches += 1
            quality_sum_oracle += oracle_mode['quality']
            quality_sum_est += est_mode['quality']

            if total_latency(oracle_mode, true_bw) <= deadline:
                deadline_met_oracle += 1
            if total_latency(est_mode, true_bw) <= deadline:
                deadline_met_est += 1

            if prev_mode is not None and est_mode['name'] != prev_mode:
                mode_switches += 1
            prev_mode = est_mode['name']

        n = len(trace)
        errors = np.array(errors)
        results[f'alpha_{alpha}'] = {
            'alpha': alpha,
            'mape': round(float(np.mean(np.abs(errors / trace))), 4),
            'mae_mbps': round(float(np.mean(np.abs(errors))), 2),
            'mode_match_rate': round(mode_matches / n, 4),
            'quality_gap': round((quality_sum_oracle - quality_sum_est) / n, 2),
            'oracle_deadline_rate': round(deadline_met_oracle / n, 4),
            'est_deadline_rate': round(deadline_met_est / n, 4),
            'mode_switches': mode_switches,
            'switch_rate': round(mode_switches / n, 4),
        }

    return results


# =========================================================================
# Experiment 4: Conservative Factor Sensitivity
# =========================================================================

def exp_conservative_factor(num_steps=5000, alpha=0.3):
    """Test sensitivity to conservative factor."""
    factors = [0.6, 0.7, 0.8, 0.9, 1.0]
    scenarios = ['urban_mmwave', 'vehicular']
    deadline = 2000

    results = {}
    for scenario in scenarios:
        scenario_results = {}
        for cf in factors:
            trace_gen = SyntheticLumos5GTrace(scenario, seed=42)
            trace = trace_gen.generate_trace(num_steps)
            estimator = EWMAEstimator(alpha=alpha, conservative_factor=cf)

            quality_sum = 0
            deadline_met = 0

            for true_bw in trace:
                est_bw = estimator.update(true_bw)
                est_mode = select_mode_estimated(est_bw, deadline)
                quality_sum += est_mode['quality']
                if total_latency(est_mode, true_bw) <= deadline:
                    deadline_met += 1

            n = len(trace)
            scenario_results[f'cf_{cf}'] = {
                'conservative_factor': cf,
                'avg_quality': round(quality_sum / n, 2),
                'deadline_rate': round(deadline_met / n, 4),
            }
        results[scenario] = scenario_results

    return results


# =========================================================================
# Experiment 5: Robustness Under Sudden Drops (mmWave blockage)
# =========================================================================

def exp_sudden_drops(num_steps=5000, alpha=0.3, cf=0.8):
    """Test protocol behavior during sudden bandwidth drops."""
    # Generate urban mmWave trace with injected sudden drops
    trace_gen = SyntheticLumos5GTrace('urban_mmwave', seed=42)
    trace = trace_gen.generate_trace(num_steps)

    # Inject 10 sudden blockage events (drop to 2 Mbps for 5 steps)
    rng = np.random.RandomState(123)
    blockage_starts = sorted(rng.choice(range(100, num_steps - 50), size=10, replace=False))
    for start in blockage_starts:
        trace[start:start+5] = 2.0  # mmWave blockage

    deadline = 2000
    estimator = EWMAEstimator(alpha=alpha, conservative_factor=cf)

    # Track behavior around blockage events
    results = {'blockage_events': []}

    for event_idx, start in enumerate(blockage_starts):
        window = range(max(0, start - 5), min(num_steps, start + 15))
        event_data = []

        # Reset estimator state to realistic pre-blockage
        est = EWMAEstimator(alpha=alpha, conservative_factor=cf)
        # Warm up with trace before blockage
        for i in range(max(0, start - 50), start - 5):
            est.update(trace[i])

        for i in window:
            true_bw = trace[i]
            est_bw = est.update(true_bw)
            oracle_mode = select_mode_oracle(true_bw, deadline)
            est_mode = select_mode_estimated(est_bw, deadline)
            event_data.append({
                'step': int(i - start),  # Relative to blockage start
                'true_bw': round(float(true_bw), 1),
                'est_bw': round(float(est_bw), 1),
                'oracle_mode': oracle_mode['name'],
                'est_mode': est_mode['name'],
                'oracle_meets': total_latency(oracle_mode, true_bw) <= deadline,
                'est_meets': total_latency(est_mode, true_bw) <= deadline,
            })

        results['blockage_events'].append(event_data)

    # Summary stats
    total_blockage_steps = sum(len(e) for e in results['blockage_events'])
    est_deadline_misses = sum(
        1 for event in results['blockage_events']
        for step in event
        if not step['est_meets']
    )
    oracle_deadline_misses = sum(
        1 for event in results['blockage_events']
        for step in event
        if not step['oracle_meets']
    )

    # Recovery time: how many steps after blockage ends until estimator recovers
    recovery_times = []
    for event in results['blockage_events']:
        blockage_end = None
        recovery_step = None
        for step in event:
            if step['step'] >= 5 and blockage_end is None:  # After blockage
                blockage_end = step['step']
            if blockage_end is not None and step['oracle_mode'] == step['est_mode']:
                recovery_step = step['step']
                break
        if blockage_end is not None and recovery_step is not None:
            recovery_times.append(recovery_step - blockage_end)

    results['summary'] = {
        'total_blockage_window_steps': total_blockage_steps,
        'est_deadline_misses_in_window': est_deadline_misses,
        'oracle_deadline_misses_in_window': oracle_deadline_misses,
        'avg_recovery_steps': round(float(np.mean(recovery_times)), 1) if recovery_times else None,
        'max_recovery_steps': int(max(recovery_times)) if recovery_times else None,
    }

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("Bandwidth Estimation Validation Experiment")
    print("=" * 70)

    all_results = {
        'metadata': {
            'experiment': 'bw_estimation_validation',
            'timestamp': datetime.now().isoformat(),
            'num_steps': 5000,
            'operating_modes': [m['name'] for m in OPERATING_MODES],
            'model': 'Qwen-7B (primary)',
        }
    }

    # Exp 1: Estimation error
    print("\n[1/5] Estimation Error Distribution...")
    all_results['estimation_error'] = exp_estimation_error()
    for sc, r in all_results['estimation_error'].items():
        re = r['relative_error']
        print(f"  {sc:20s}: MAPE={re['mape']:.1%}, median_rel_err={re['median']:.1%}, "
              f"underestimate_rate={r['underestimate_rate']:.1%}")

    # Exp 2: Mode selection accuracy
    print("\n[2/5] Mode Selection Accuracy vs Oracle...")
    all_results['mode_selection'] = exp_mode_selection_accuracy()
    for sc, deadlines in all_results['mode_selection'].items():
        d2s = deadlines.get('d2000ms', {})
        print(f"  {sc:20s} @2s: match={d2s.get('exact_match_rate',0):.1%}, "
              f"q_gap={d2s.get('quality_gap',0):.1f}%, "
              f"deadline: oracle={d2s.get('oracle_deadline_rate',0):.1%} "
              f"est={d2s.get('estimated_deadline_rate',0):.1%}")

    # Exp 3: Alpha sensitivity
    print("\n[3/5] Alpha Sensitivity (vehicular scenario, 2s deadline)...")
    all_results['alpha_sensitivity'] = exp_alpha_sensitivity()
    for key, r in all_results['alpha_sensitivity'].items():
        print(f"  alpha={r['alpha']:.1f}: MAPE={r['mape']:.1%}, mode_match={r['mode_match_rate']:.1%}, "
              f"q_gap={r['quality_gap']:.1f}%, switches={r['mode_switches']}")

    # Exp 4: Conservative factor
    print("\n[4/5] Conservative Factor Sensitivity...")
    all_results['conservative_factor'] = exp_conservative_factor()
    for sc, factors in all_results['conservative_factor'].items():
        print(f"  {sc}:")
        for key, r in factors.items():
            print(f"    cf={r['conservative_factor']:.1f}: quality={r['avg_quality']:.1f}%, "
                  f"deadline={r['deadline_rate']:.1%}")

    # Exp 5: Sudden drops
    print("\n[5/5] Sudden Blockage Recovery...")
    all_results['sudden_drops'] = exp_sudden_drops()
    s = all_results['sudden_drops']['summary']
    print(f"  Deadline misses (est): {s['est_deadline_misses_in_window']} / {s['total_blockage_window_steps']}")
    print(f"  Deadline misses (oracle): {s['oracle_deadline_misses_in_window']} / {s['total_blockage_window_steps']}")
    print(f"  Avg recovery: {s['avg_recovery_steps']} steps, Max: {s['max_recovery_steps']} steps")

    # Save results (convert numpy types for JSON)
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = Path(__file__).parent.parent / 'results' / f'exp_bw_estimation_{ts}.json'
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {outfile}")

    # Print summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER (Table: BW Estimation Robustness)")
    print("=" * 70)
    print(f"{'Scenario':<20} {'MAPE':>8} {'Mode Match':>12} {'Q Gap':>8} {'DL (est)':>10} {'DL (oracle)':>12}")
    print("-" * 70)
    for sc in ['urban_mmwave', 'urban_sub6', 'suburban_mixed', 'vehicular']:
        err = all_results['estimation_error'][sc]
        ms = all_results['mode_selection'][sc].get('d2000ms', {})
        print(f"{sc:<20} {err['relative_error']['mape']:>7.1%} "
              f"{ms.get('exact_match_rate',0):>11.1%} "
              f"{ms.get('quality_gap',0):>7.1f}% "
              f"{ms.get('estimated_deadline_rate',0):>9.1%} "
              f"{ms.get('oracle_deadline_rate',0):>11.1%}")

    return all_results


if __name__ == '__main__':
    main()
