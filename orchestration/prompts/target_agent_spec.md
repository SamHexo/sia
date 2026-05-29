## Target Agent Technical Specification

Every target_agent.py must conform to the following requirements.

### Required CLI arguments

```
--dataset_dir               Absolute path to the dataset directory (READ-ONLY)
--working_dir               Absolute path to gen_N/workspace/ — pure scratch space (READ-WRITE, not carried forward)
--solutions_dir             Absolute path to gen_N/solutions/ — write evaluated solution files here
--exit_reason_path          Absolute path to gen_N/exit_reason.txt — write exit reason on any exit
--agent_execution_path      Absolute path to gen_N/agent_execution.json — write execution trace here
--supervision_log_path      Absolute path to gen_N/supervision_log.txt — append supervision decisions here
--task_model_logs_dir       Absolute path to gen_N/task_model_logs/ — write per-turn LLM call logs here
--shared_dir                Absolute path to tasks/_shared/ (add to sys.path to import utilities)
--model                     Model name for all LLM calls — never hardcode this
--task_model_temperature    Sampling temperature for call_task_model() calls (float)
--state_file                Absolute path to gen_N/state.json
--current_gen               Current generation number (int)
--supervision_model         Model for supervision calls (cheap model)
--supervision_interval      Run supervision every N outer iterations (int)
--supervision_check_timeout Seconds without a supervision heartbeat before broken_gen (int)
--gen0_evolve_duration      Gen-0 only: EVOLVE after this many seconds (int, 0 = disabled)
--exp_duration_seconds      Total experiment budget in seconds — use to compute remaining time for supervision
```

There is **no `--max_turns` or `--target_agent_timeout`**. Stopping is handled entirely
by the supervision system and the global safety timer in the orchestrator.

### File access

- `--dataset_dir`: READ-ONLY
- `--working_dir`: READ-WRITE — pure scratch space; use freely for experiments, logs, temp files (not carried forward to the next generation)
- `--solutions_dir`: READ-WRITE — write evaluated solution files here; filename **must** be `{uid}.py` (8 lowercase hex chars, e.g. `a3f2b1c0.py`). The `write_file` tool enforces this pattern and returns an explicit error if it is not met.
- `--exit_reason_path`: WRITE — write exit reason string here on any exit
- `--agent_execution_path`: WRITE — write agent_execution.json here; overwrite after each iteration
- `--supervision_log_path`: WRITE — append one line per supervision decision
- `--task_model_logs_dir`: WRITE — write per-turn LLM call logs here (directory, created by orchestrator)

### Solution archival

Each evaluated solution **must** be registered in state.json via `write_tree_node`:

```python
from utils.tree_utils import write_tree_node

# Allocate path in solutions_dir (scaffold controls the UID):
candidate_uid  = str(uuid.uuid4())[:8]
candidate_path = os.path.join(solutions_dir, f"{candidate_uid}.py")
# ... inner LLM writes solution to candidate_path ...
node_id = write_tree_node(state_file, candidate_path, result_dict, current_gen)
```

`result_dict` must contain at minimum:
- `"score"` (float)
- `"lower_is_better"` (bool) — from evaluate.py output

`iteration_id` is computed automatically by `write_tree_node` (sequential per generation).

### Evaluation

The scaffold runs evaluate.py directly via subprocess — the inner LLM never calls it:

```python
proc = subprocess.run(
    [sys.executable, evaluate_path, candidate_path],
    capture_output=True, text=True, cwd=working_dir, timeout=300,
)
full_out = proc.stdout + proc.stderr
# Parse result:
json_str = full_out.split("RESULT_JSON:", 1)[1].split("\n", 1)[0].strip()
result = json.loads(json_str)
```

evaluate.py always prints `RESULT_JSON:{...}` as its last line.

### Supervision (required)

After each MCTS iteration (outer loop), call check_supervision every
`--supervision_interval` iterations:

```python
sys.path.insert(0, args.shared_dir)
from supervision_call import check_supervision

sup = check_supervision(
    tree_summary=summarize_tree(args.state_file),
    elapsed_seconds=time.time() - start_time,
    remaining_seconds=args.supervision_check_timeout - (time.time() - last_supervision_time),
    current_gen=args.current_gen,
    model=args.supervision_model,
    gen0_evolve_duration_s=args.gen0_evolve_duration,
)
decision, reason = sup["decision"], sup["reason"]
# decision is one of: "continue", "evolve", "stop", "fail"
```

Write the exit reason to `--exit_reason_path`:
- `"broken_gen: <reason>"` — generation failed
- `"evolve: <reason>"` — normal completion, meta-agent creates next gen
- `"stop: <reason>"` — terminate the run

Also implement a heartbeat timeout for gen > 0: if no successful supervision check
in `--supervision_check_timeout` seconds, write `broken_gen: heartbeat timeout`.

### Execution log

Write the outer loop trajectory to `--agent_execution_path` (agent_execution.json).
**Update it after every iteration** (overwrite in place) so it survives a crash.

Minimum useful fields per record:

```json
[
  {
    "iteration": 1,
    "elapsed_s": 42.1,
    "candidate_uid": "a3f2b1c0",
    "parent_score": 0.6092,
    "solution_written": true,
    "eval_score": 0.6134,
    "eval_error": null,
    "node_id": "node_0002",
    "inner_trajectory": [
      {"turn": 1, "tool_calls": [...], "content": "..."},
      {"turn": 2, "tool_calls": [...], "content": "..."}
    ]
  }
]
```

Add any fields relevant to your search strategy (e.g. PUCT scores, parent selection, branch tags). The orchestrator reads this file for diagnostics but does not enforce a schema — any valid JSON list is accepted.
