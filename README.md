# SIA — Self-Improving Agent

SIA is a framework for building self-improving AI systems that autonomously evolve their search strategy across generations on scientific tasks.

## Results

<table width="100%">
  <tr>
    <td width="50%" align="center"><br><img src="plots/gpqa.png" alt="GPQA Results" height="220"></td>
    <td width="50%" align="center"><br><img src="plots/ml_agent.png" alt="ML Agent Results" height="220"></td>
  </tr>
</table>

<p align="center"><i>Progressive improvement across generations on scientific tasks.</i></p>

---

## Overview

<p align="center"><img src="plots/flow.png" alt="SIA orchestration flow" width="720"></p>

SIA runs a loop of two agents:

- **Meta-agent** — reads the task, previous execution history, and persistent search state, then writes or improves the target-agent scaffold.
- **Target agent** — executes the task by running a search loop, evaluating candidate solutions, and writing results into a persistent tree state.

Each iteration is called a **generation**. The search state accumulates across all generations: the meta-agent improves the *strategy*, not the *search position*.

---

## Architecture

### Core principle

> **The scaffold evolves across generations. The search tree persists across generations. Generations do not restart the task.**

Passing only the best solution between generations collapses search into single-trajectory hill-climbing. SIA instead preserves the **full evaluation tree** — all nodes, scores, failed branches, and partial discoveries — so future generations can:

- Continue from any existing node, not just the best
- Avoid re-exploring already-visited regions
- Adapt explore/exploit balance based on accumulated evidence
- Branch into unexplored directions after local optima

### Generation folders

Each generation lives in its own directory inside the run folder:

```
runs/run_{id}/
├── venv/                          # Isolated Python env (auto-created)
├── gen_0/                         # Reference agent (copied verbatim, no meta-agent call)
│   ├── target_agent/
│   │   ├── target_agent.py        # Main search scaffold
│   │   ├── conf.yaml              # Agent search config (PUCT, exploration params, etc.)
│   │   └── utils/                 # Shared helpers (tree_utils, etc.)
│   ├── workspace/                 # Pure scratch space (not carried forward)
│   ├── state.json                 # Persistent search tree (copied forward each gen)
│   ├── solutions/                 # Candidate solutions keyed by UUID (copied forward)
│   ├── agent_execution.json       # Full turn-by-turn trajectory
│   ├── supervision_log.txt        # Append-only supervision decision history
│   ├── task_model_logs/           # Per-turn LLM call logs
│   ├── target_agent_stdout.log    # Combined stdout/stderr of target agent process
│   └── context.md                 # Cumulative run history
├── gen_1/                         # Meta-agent creates new scaffold; copies state from gen_0
│   ├── target_agent/              # Modified by meta-agent
│   ├── state.json                 # Copied from gen_0, extended by gen_1's agent
│   ├── improvement.md             # Meta-agent's analysis and plan for this gen
│   └── ...
├── gen_N/                         # Each gen copies gen_{N-1} forward (exit_reason.txt never copied)
├── gen_broken/                    # Failed gens quarantined here on broken_gen
│   └── gen_2_20260528_143200/
└── private_scores/                # Private eval results per gen
    └── gen_N/
        └── private_result.json
```

### State.json — the persistent tree

`state.json` is created before gen_0 and **copied forward** to each new generation. The target agent reads it at startup, continues the search from existing nodes, and appends new evaluated nodes.

Schema (v0.2):

```json
{
  "version": "0.2",
  "task_id": "denoising",
  "root_id": "root",
  "best_node_id": "node_0003",
  "nodes": {
    "root": {
      "id": "root", "parent": null, "children": ["node_0000"],
      "uuid": "root", "generation": 0, "result": null, "visits": 0, "status": "root"
    },
    "node_0003": {
      "id": "node_0003", "parent": "root", "children": [],
      "uuid": "a1b2c3d4-...",
      "generation": 1,
      "solution_path": "gen_1/solutions/a1b2c3d4-....py",
      "result": { "score": 0.6812, "lower_is_better": false, "mse": 0.12, ... },
      "visits": 1, "status": "evaluated"
    }
  }
}
```

**Required node keys** (the orchestrator validates these):
- `uuid` — unique identifier, used to key the solution file
- `generation` — which generation produced this node (int)
- `result` — dict with at least:
  - `score` (float)
  - `lower_is_better` (bool)
  - `iteration_id` (int) — sequential index of this evaluation within the generation (0, 1, 2, …)

### Supervision system

Stopping is driven by a supervision agent running **inside** the target agent, not by a fixed turn/time limit.

Every `--supervision_interval` turns, the target agent calls `check_supervision()`, which asks a cheap LLM to decide:

| Decision | Action |
|----------|--------|
| `CONTINUE` | Keep running — scores still improving |
| `EVOLVE` | Stop this generation — meta-agent will improve the scaffold |
| `STOP` | Terminate the entire run — goal met or budget exhausted |

The target agent writes `exit_reason.txt` in `STATUS: reason` format:
- `evolve: <reason>` — generation completed its budget; meta-agent creates the next scaffold
- `stop: <reason>` — terminate the entire run
- `broken_gen: <reason>` — generation failed (agent error, heartbeat timeout, etc.)

**Gen-0 special case**: no LLM call. Gen-0 auto-EVOLVEs after `--gen0_evolve_duration` seconds — just enough time to establish a baseline.

**Heartbeat safety**: if no successful supervision call is recorded within `--supervision_check_timeout_min` minutes, the target agent writes `broken_gen` automatically. Two consecutive supervision call failures also trigger `broken_gen`.

**Retry logic**: supervision calls retry 3× with 5 s / 10 s / 20 s backoff before failing.

### Safety timer

The orchestrator starts a background timer at `1.3 × exp_duration_min` seconds. When it fires:
1. Kills the current target-agent subprocess
2. Finds the best solution across all valid generations
3. Runs final private scoring (if configured)
4. Exits cleanly

### Fallback on broken_gen

When a generation fails (`broken_gen`):
1. The broken `gen_N/` directory is moved to `gen_broken/gen_N_<timestamp>/`
2. A fresh copy of `gen_{N-1}/` is created as `gen_N/`
3. The meta-agent is re-run with a **fallback warning** that includes: the error message, the broken `target_agent.py`, the last 2000 chars of stdout, and the state validation failure reason
4. The meta-agent is explicitly told not to replicate what broke

---

## Denoising Task

### What the task is

Given raw scRNA-seq count data `X` (cells × genes), produce a denoised version with the same shape. The denoised output must be non-negative.

```python
def custom_denoise(X, **kwargs) -> np.ndarray:
    # X: (n_cells, n_genes) raw counts
    # return: (n_cells, n_genes) denoised counts, non-negative
    ...
```

### Evaluation

Two metrics, both computed in normalized space, combined into a single score in [0, 1] (higher is better):

| Metric | Baseline (identity) | Perfect |
|--------|---------------------|---------|
| MSE (log-normalized) | 0.304721 | 0.0 |
| Poisson NLL | 0.257575 | 0.031739 |

```
score = ( mse_norm + poisson_norm ) / 2
```

`lower_is_better: false` — the score is already normalized so that higher = better denoising.

### Baseline

MAGIC (graph diffusion) achieves ~0.61 on the public (pancreas) dataset and is the baseline to beat. The goal is a **novel algorithm** that outperforms MAGIC — not a tuned version of it.

```python
import magic
def custom_denoise(X, **kwargs):
    return magic.MAGIC().fit_transform(X)  # baseline — beat this
```

### Generalization

The **private test set uses different tissues** (PBMC, Tabula Muris) than the public dev set (pancreas). Methods that overfit to pancreas statistics will fail on private eval. Prefer parameter-free or self-adapting methods (graph diffusion, low-rank approximation, PCA-based smoothing) over methods trained on the public data.

### Running the denoising task

```bash
cd tasks/denoising
bash download_data.sh          # download data once

cd ../..
bash tasks/denoising/launch.sh
```

Key parameters (all have defaults in `launch.sh`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--exp_duration_min` | 60 | Total budget in minutes; safety kill at 1.3× |
| `--gen0_evolve_duration_min` | 15 | Gen-0 auto-EVOLVE after N minutes |
| `--meta_model` | `gemini/gemini-3.1-pro-preview` | Meta-agent model |
| `--task_model` | *(Tinker checkpoint)* | Target-agent model |
| `--supervision_model` | `gemini/gemini-3.1-pro-preview` | Supervision model |
| `--supervision_interval` | 5 | Supervision every N turns |
| `--supervision_check_timeout_min` | 5 | Minutes without a heartbeat before broken_gen |
| `--meta_agent_max_turns` | 100 | Turn budget for meta-agent's own loop |
| `--backend` | `openhands` | `claude` or `openhands` |
| `--run_id` | 1 | Unique run identifier |

---

## Setup

### Prerequisites

- Python 3.11+
- API keys for the models you use

```bash
export ANTHROPIC_API_KEY="..."   # Claude models / Claude Code backend
export GOOGLE_API_KEY="..."      # Gemini models
export OPENAI_API_KEY="..."      # OpenAI models
```

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The orchestrator auto-creates a **per-run venv** (`runs/run_{id}/venv/`) and installs task-specific requirements before gen_0. You don't need to install task dependencies manually.

---

## Directory Structure

```
sia/
├── orchestration/
│   ├── orchestrator.py          # Main loop: gen folders, supervision, safety timer, fallback
│   ├── context_manager.py       # Tracks scores and evolution across generations
│   ├── plot_utils.py            # Private score plots at end of run
│   └── prompts/
│       ├── meta_agent_prompt.md          # Meta-agent prompt template
│       ├── supervision_prompt.md         # Supervision decision prompt
│       ├── scaffold_design_guidelines.md # Design philosophy injected into meta-agent prompt
│       └── target_agent_spec.md          # Target-agent interface spec
├── tasks/
│   ├── _shared/
│   │   ├── call_task_model.py   # Unified LLM caller (Tinker + litellm)
│   │   ├── supervision_call.py  # check_supervision() with retry + gen-0 fast-path
│   │   └── tools.py             # Sandboxed bash / read_file / write_file
│   └── denoising/
│       ├── data/
│       │   ├── public/
│       │   │   └── evaluate.py   # Public evaluator (prints RESULT_JSON: to stdout)
│       │   └── private/
│       │       └── evaluate.py   # Private evaluator (never exposed to target agent)
│       ├── reference/
│       │   ├── task.md           # Task description (read by meta-agent and target agent)
│       │   └── reference_target_agent/
│       │       ├── target_agent.py  # Reference scaffold
│       │       ├── conf.yaml        # Default search config
│       │       ├── README.md        # Descriptive state snapshot of the current scaffold
│       │       └── utils/
│       │           └── tree_utils.py  # write_tree_node, read_state, get_best_node, ...
│       ├── requirements.txt
│       ├── download_data.sh
│       └── launch.sh
└── runs/                         # Generated during execution (gitignored)
    └── run_{id}/
        ├── venv/
        ├── gen_0/ … gen_N/
        ├── gen_broken/
        └── private_scores/
```

---

## Adding a New Task

Only three files are required:

```
tasks/{task-id}/
├── data/
│   ├── public/
│   │   └── evaluate.py    ← public evaluator
│   └── private/
│       └── evaluate.py    ← private evaluator (never exposed to target agent)
└── reference/
    ├── task.md            ← task description (read by meta-agent and target agent)
    └── reference_target_agent/
        ├── target_agent.py  ← reference scaffold
        ├── conf.yaml        ← agent search config
        └── README.md        ← descriptive state snapshot (no instructions)
```

### `reference/task.md`

Injected verbatim into the target agent's prompt. Include:
- What the task is and what the solution function must do
- Data format and directory layout (tell the agent exactly which files exist)
- Scoring metric and direction (higher vs. lower is better)
- Any domain constraints

> **Tip:** Explicitly list `{dataset_dir}/` contents so the agent doesn't waste turns on `ls` / `find`. Tell it: CWD is `{working_dir}`, use absolute paths for `{dataset_dir}`, don't explore beyond these two directories.

### `data/public/evaluate.py`

Called by the target agent to score its solution. Must:
1. Accept the solution path as `sys.argv[1]` (optionally `--iteration_id N`)
2. Print `RESULT_JSON:{...}` to stdout — this is how the orchestrator and target agent read the score
3. The JSON must contain at minimum:
   ```json
   { "score": 0.87, "lower_is_better": false, "iteration_id": 0, "error": null }
   ```
   - `score` — float in a consistent range (e.g. 0–1)
   - `lower_is_better` — bool, fixed for this task, never changes
   - `iteration_id` — int, sequential index of this evaluation within the current generation

### `data/private/evaluate.py`

Called by the orchestrator only, never exposed to the target agent. Same interface as the public evaluator (accept solution path as argv[1], print `RESULT_JSON:`). The orchestrator runs it in a separate `private_scores/` directory so the meta-agent cannot read the result.

### `reference/reference_target_agent/`

The initial scaffold shown verbatim to the meta-agent. Must implement the target-agent contract below.

---

## Target-Agent Contract

Any target agent written by the meta-agent must:

1. **Accept these CLI args** (passed by the orchestrator):
   ```
   --dataset_dir, --working_dir, --solutions_dir,
   --exit_reason_path, --agent_execution_path,
   --supervision_log_path, --task_model_logs_dir,
   --shared_dir, --model, --task_model_temperature,
   --state_file, --current_gen,
   --supervision_model, --supervision_interval, --supervision_check_timeout,
   --gen0_evolve_duration
   ```

2. **Run a `while True` loop** — no fixed turn limit. Stopping is handled by supervision.

3. **Call `check_supervision()`** every `--supervision_interval` turns — returns `{"decision": ..., "reason": ...}`. On `"evolve"` write `evolve: <reason>` to `exit_reason.txt` and break; on `"stop"` write `stop: <reason>` and break; on error write `broken_gen: <reason>` and break.

4. **Write evaluated solutions to `state.json`** via `write_tree_node(state_file, solution_path, result_dict, current_gen)`. The result dict comes directly from `evaluate.py` — `score`, `lower_is_better`, and `iteration_id` are guaranteed to be present.

5. **Write each candidate to a distinct file path** — do not overwrite. `write_tree_node` archives it to `solutions/{uuid}.py` immediately.

6. **Write `agent_execution.json`** after every turn (full trajectory for meta-agent context).

### Shared utilities

```python
# Available after sys.path.insert(0, args.shared_dir)
from call_task_model import call_task_model     # unified LLM caller
from supervision_call import check_supervision  # CONTINUE / EVOLVE / STOP / fail
import tools                                    # sandboxed bash / read_file / write_file

# Available in target_agent/utils/ (or copy to your own utils/)
from utils.tree_utils import (
    write_tree_node,      # write a node to state.json
    read_state,           # load state.json
    get_best_node,        # (node_dict, score) — best across all gens
    get_generation_nodes, # list of nodes from a specific generation
    summarize_tree,       # compact summary string (raises ValueError on broken state)
)
```

### `conf.yaml`

Agent-specific search configuration — things the meta-agent can tune each generation:

```yaml
search:
  n_candidates: 3
  c_puct: 1.5
  branching_factor: 2
  max_depth: 5

exploration:
  temperature: 1.0
  top_k_parents: 3
  restart_on_stagnation: true
  stagnation_patience: 5
```

`lower_is_better` is a task property fixed in `evaluate.py` — it does not belong in `conf.yaml`.

---

## CLI Reference

### Orchestrator (`orchestration/orchestrator.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_dir` | *(required)* | Path to task directory |
| `--run_id` | 1 | Unique run identifier |
| `--backend` | `claude` | `claude` (Claude Code SDK) or `openhands` |
| `--meta_model` | `gemini/gemini-3.1-pro-preview` | Meta-agent model |
| `--task_model` | `claude-haiku-4-5-20251001` | Target-agent model |
| `--exp_duration_min` | 60 | Total budget (minutes); safety kill at 1.3× |
| `--task_model_temperature` | 0.3 | Target-agent sampling temperature |
| `--supervision_model` | `gemini/gemini-3.1-pro-preview` | Supervision model |
| `--supervision_interval` | 5 | Supervision every N turns |
| `--supervision_check_timeout_min` | 5 | Minutes before heartbeat triggers broken_gen |
| `--gen0_evolve_duration` | 900 | Gen-0 auto-EVOLVE after N seconds |
| `--meta_agent_max_turns` | 100 | Turn budget for meta-agent's tool loop |
| `--private_scores_task_models` | `` | Comma-separated models for private eval |

### Backends

**Claude Code** (default) — Claude models only, uses the Claude Agent SDK:
```bash
--backend claude --meta_model haiku   # haiku / sonnet / opus
```

**OpenHands** — any LLM via litellm:
```bash
--backend openhands --meta_model "gemini/gemini-3.1-pro-preview"
--backend openhands --meta_model "openai/gpt-4o"
--backend openhands --meta_model "anthropic/claude-sonnet-4-6"
```

---

## Troubleshooting

**Run directory already exists**
Use a different `--run_id` or delete the existing run: `rm -rf runs/run_1`.

**Target agent exits immediately / supervision heartbeat timeout**
Check `gen_N/supervision_log.txt`, `gen_N/exit_reason.txt`, and `gen_N/target_agent_stdout.log`. Usually caused by a broken `state.json` (missing required keys) or a supervision model API error. The fallback mechanism will quarantine the broken gen and retry.

**`broken_gen` loops / repeated fallbacks**
Check `gen_broken/` for the broken agents and their stdout logs. The meta-agent receives the broken agent code and error details — if it keeps failing, the bug is likely an import error or missing dependency. Check `gen_N/target_agent_stdout.log`.

**Private score diverges from public score**
The target agent is overfitting to the public evaluation set. For tasks with held-out test data (like denoising), use parameter-free or self-adapting methods. Avoid training on the public data.

**`ImportError` in venv**
The orchestrator creates a fresh venv per run and installs `tasks/{task}/requirements.txt`. If a package is missing: `runs/run_1/venv/bin/pip install <package>`.
