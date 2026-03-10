# Remote Operations Workflow

How to run experiments on Vast.ai GPU servers, sync results, and coordinate multiple agents.
For CLI syntax → `REF_vastai_cli.md`. For LLM SDK calls → `REF_llm_api.md`.

## GPU Selection Strategy

| Experiment type | VRAM needed | Search query |
|----------------|-------------|--------------|
| API-only (A5, I1, I3, D1, D6) | 0 GB (CPU ok) | `'reliability>0.95 num_gpus=1' -o 'dph+'` |
| Local inference (Ollama, llama.cpp) | 24 GB | `'gpu_ram>=24 reliability>0.95 num_gpus=1' -o 'dph+'` |
| Fine-tuning (FinDAP) | 40+ GB | `'gpu_ram>=40 reliability>0.95 num_gpus>=2' -o 'dph+'` |

Rule: if your local machine will be off or busy, rent even for API-only experiments — a cheap CPU instance ($0.05–0.15/hr) keeps runs going 24/7.

## Experiment Dispatch

| Where | Experiments | Why |
|-------|------------|-----|
| Local | Short runs (<30 min), figure generation, LaTeX builds | Free, instant |
| Remote (API-only) | A1, A5, I1, I2, I3, D1, D6 with N>100 | Runs overnight unattended |
| Remote (GPU) | Ollama inference, FinDAP training | Needs VRAM |

## Environment Setup

On the remote server, create and run a `setup.sh`:

```bash
#!/bin/bash
# setup.sh — run once after instance creation
apt-get update && apt-get install -y tmux
pip install openai anthropic google-genai requests python-dotenv tqdm pydantic

# Log what's installed for any agent to verify
echo "Setup completed: $(date)" > /workspace/SETUP_LOG.txt
pip freeze >> /workspace/SETUP_LOG.txt
nvidia-smi >> /workspace/SETUP_LOG.txt 2>/dev/null || echo "No GPU" >> /workspace/SETUP_LOG.txt
```

Upload project code and API keys:

```bash
vastai copy /local/path/CFA_essay/ <ID>:/workspace/CFA_essay/
vastai copy /local/path/.env <ID>:/workspace/CFA_essay/.env
```

## Checkpointing Protocol

All batch experiments MUST save incrementally. Never hold all results in memory until the end.

```python
# Save every CHECKPOINT_INTERVAL items
CHECKPOINT_INTERVAL = 25  # adjust: 10 for slow experiments, 50 for fast

checkpoint = {"completed": 0, "total": N, "results": []}
checkpoint_path = results_dir / "results_partial.json"

for i, item in enumerate(items):
    result = run_single(item)
    checkpoint["results"].append(result)
    checkpoint["completed"] = i + 1
    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        json.dump(checkpoint, open(checkpoint_path, "w"), indent=2)

# Final: rename partial → final
final_path = results_dir / "results.json"
json.dump(checkpoint, open(final_path, "w"), indent=2)
checkpoint_path.unlink(missing_ok=True)
```

Resume from checkpoint:

```python
if checkpoint_path.exists():
    checkpoint = json.load(open(checkpoint_path))
    done = checkpoint["completed"]
    items = items[done:]  # skip already-completed
```

## Sync Protocol

```bash
# Upload code (local → remote)
vastai copy ./CFA_essay/ <ID>:/workspace/CFA_essay/

# Pull results (remote → local)
vastai copy <ID>:/workspace/CFA_essay/experiments/A5_option_bias/results/ ./experiments/A5_option_bias/results/

# Quick progress check (no SSH needed)
vastai execute <ID> 'python3 -c "
import json, glob
for f in glob.glob(\"/workspace/CFA_essay/experiments/*/results/*partial*\"):
    d = json.load(open(f))
    print(f\"{f}: {d[\"completed\"]}/{d[\"total\"]}\")"'
```

Always use `tmux` on the server so experiments survive SSH disconnect:

```bash
ssh $(vastai ssh-url <ID>)
tmux new -s exp
cd /workspace/CFA_essay
python -m experiments.A5_option_bias.run_experiment --dataset easy --model gpt-4o-mini
# Ctrl+B, D to detach — experiment keeps running
```

Sync frequency: pull results every 30–60 min during active runs.

## Multi-Agent Coordination

When multiple agents share one server, use a status file:

```bash
# /workspace/AGENT_STATUS.json
[
  {"id": "agent-1", "experiment": "A5_option_bias", "gpu_gb": 0, "status": "running", "started": "2026-02-11T10:00:00Z"},
  {"id": "agent-2", "experiment": "I1_counterfactual", "gpu_gb": 0, "status": "running", "started": "2026-02-11T10:05:00Z"}
]
```

Rules:
1. **Read before starting** — `cat /workspace/AGENT_STATUS.json`
2. **Register yourself** — append your entry before launching
3. **API-only experiments** (gpu_gb=0) — always OK to run concurrently
4. **GPU experiments** — sum `gpu_gb` of running entries, compare to total VRAM (`nvidia-smi`), only start if headroom exists
5. **Update when done** — set `status: "completed"` or remove your entry

## Cost Management

Decision tree:

```
More experiments planned within 24h?
├── YES → `vastai stop instance <ID>`    (keeps data, ~$0.01–0.05/hr storage)
└── NO  → Results synced to local?
          ├── YES → `vastai destroy instance <ID>`  (zero cost, irreversible)
          └── NO  → Sync first, then destroy
```

- `stop` = pause billing for GPU, pay only storage. Data intact. Resume anytime.
- `destroy` = delete everything. Cannot recover. Re-create from scratch is cheap (~2 min).
- Rule: idle >24h with no planned work → destroy. Re-creation cost < accumulated storage fees.
