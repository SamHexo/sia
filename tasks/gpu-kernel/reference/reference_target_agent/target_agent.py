#!/usr/bin/env python3
"""
Reference target agent: MCTS outer loop + focused inner multi-turn LLM per node.

Architecture:
  outer loop — MCTS (PUCT selection → expand → evaluate → write node)
  inner loop — short multi-turn LLM whose sole goal is to write ONE solution file

Usage:
    python target_agent/target_agent.py \
        --dataset_dir        /path/to/data/public \
        --working_dir        /path/to/gen_N/workspace \
        --solutions_dir      /path/to/gen_N/solutions \
        --exit_reason_path   /path/to/gen_N/exit_reason.txt \
        --agent_execution_path /path/to/gen_N/agent_execution.json \
        --shared_dir         /path/to/tasks/_shared \
        --model              openai/gpt-oss-120b
"""

import argparse
import datetime
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import uuid as _uuid_mod
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.tree_utils import write_tree_node as _write_tree_node_util, summarize_tree


TOOLS = [
    {
        "name": "bash",
        "description": "Run a bash command inside the working or dataset directory.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the working, solutions, or dataset directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a scratch file inside the working directory. Use submit_solution to submit a candidate solution.",
        "parameters": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Path inside working_dir"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "submit_solution",
        "description": (
            "Your primary feedback loop. "
            "Submit Python code as a candidate solution: the scaffold evaluates it immediately and returns {uid, score, error}. "
            "Expected workflow: write an attempt → submit → read the score → improve → submit again. "
            "There is no separate evaluate tool — this is the only way to get a score. "
            "Call as many times as needed. When you stop making tool calls, the run ends."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Complete Python source for your solution (function signature is in the task description)"},
            },
            "required": ["code"],
        },
    },
]

# TypeScript-syntax tool definitions for the Harmony developer message
TOOLS_TS = """\
namespace functions {

// Run a bash command inside the working or dataset directory.
type bash = (_: {
command: string,
}) => any;

// Read a file from the working, solutions, or dataset directory.
type read_file = (_: {
path: string,
}) => any;

// Write a scratch file inside the working directory only.
type write_file = (_: {
path: string,
content: string,
}) => any;

// Primary feedback loop — submit code, get score immediately.
// No separate evaluate tool exists: this is the only way to measure performance.
// Expected workflow: write attempt → submit → read score → improve → submit again.
// Call as many times as needed. Stop only when satisfied.
type submit_solution = (_: {
// Complete Python source for your solution
code: string,
}) => any;

} // namespace functions"""


def _puct_score(node: dict, total_visits: int, c_puct: float) -> float:
    """UCB1-style PUCT score for node selection (exploitation + exploration)."""
    q = float((node.get("result") or {}).get("score", 0.0))
    n = max(node.get("visits", 1), 1)
    return q + c_puct * math.sqrt(math.log(max(total_visits, 1)) / n)


def _select_parent(state_file: str, c_puct: float) -> tuple[dict | None, str | None]:
    """Select the best node to expand via PUCT. Returns (node, code_str) or (None, None)."""
    try:
        with open(state_file) as f:
            tree = json.load(f)
    except Exception:
        return None, None

    nodes = tree.get("nodes", {})
    root_id = tree.get("root_id")
    evaluated = [
        (nid, n) for nid, n in nodes.items()
        if isinstance(n.get("result"), dict) and "score" in n["result"]
        and nid != root_id and n.get("status") != "buggy"
    ]
    if not evaluated:
        return None, None

    total_visits = sum(n.get("visits", 1) for _, n in evaluated)
    best_nid, best_node = max(evaluated, key=lambda x: _puct_score(x[1], total_visits, c_puct))

    parent_code = None
    sol_path = best_node.get("solution_path")
    if sol_path and os.path.exists(sol_path):
        try:
            parent_code = Path(sol_path).read_text(encoding="utf-8")
        except Exception:
            pass

    # Increment visit count to update exploration bonus for next selection
    tree["nodes"][best_nid]["visits"] = best_node.get("visits", 1) + 1
    tmp = state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tree, f, indent=2)
    os.replace(tmp, state_file)

    return best_node, parent_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",           required=True)
    parser.add_argument("--working_dir",           required=True, help="Pure scratch space for this generation")
    parser.add_argument("--solutions_dir",         required=True, help="Path to gen_N/solutions/ — write solution files here")
    parser.add_argument("--exit_reason_path",      required=True, help="Path to gen_N/exit_reason.txt")
    parser.add_argument("--agent_execution_path",  required=True, help="Path to gen_N/agent_execution.json")
    parser.add_argument("--supervision_log_path",  required=True, help="Path to gen_N/supervision_log.txt")
    parser.add_argument("--task_model_logs_dir",   required=True, help="Path to gen_N/task_model_logs/")
    parser.add_argument("--shared_dir",            required=True, help="Path to tasks/_shared/")
    parser.add_argument("--model",                 required=True)
    parser.add_argument("--task_model_temperature", type=float, default=0.3)
    parser.add_argument("--state_file", default=None)
    parser.add_argument("--current_gen", type=int, default=0)
    parser.add_argument("--supervision_model", default=None)
    parser.add_argument("--supervision_interval", type=int, default=5)
    parser.add_argument("--supervision_check_timeout", type=int, default=300,
                        help="Seconds without a supervision heartbeat before broken_gen (default: 300 = 5 min)")
    parser.add_argument("--gen0_evolve_duration", type=int, default=0)
    parser.add_argument("--exp_duration_seconds", type=int, default=0,
                        help="Total experiment budget in seconds (used to compute remaining budget for supervision)")
    parser.add_argument("--seed_solution", default=None,
                        help="Path to a reference solution to evaluate and seed the tree before the MCTS loop")
    args = parser.parse_args()

    dataset_dir          = os.path.abspath(args.dataset_dir)
    working_dir          = os.path.abspath(args.working_dir)
    solutions_dir        = os.path.abspath(args.solutions_dir)
    exit_reason_path     = os.path.abspath(args.exit_reason_path)
    agent_execution_path = os.path.abspath(args.agent_execution_path)
    supervision_log_path = os.path.abspath(args.supervision_log_path)
    log_dir              = os.path.abspath(args.task_model_logs_dir)
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(solutions_dir, exist_ok=True)

    sys.path.insert(0, args.shared_dir)
    from call_task_model import call_task_model
    from supervision_call import check_supervision
    import tools as _tools

    # Load conf.yaml
    conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.yaml")
    c_puct = 1.5
    max_inner_turns = 12
    if os.path.exists(conf_path):
        try:
            import yaml
            conf = yaml.safe_load(open(conf_path)) or {}
            c_puct          = conf.get("search", {}).get("c_puct", c_puct)
            max_inner_turns = conf.get("search", {}).get("max_inner_turns", max_inner_turns)
        except Exception as _e:
            logger.warning(f"conf.yaml load failed: {_e}")

    MAX_OUTPUT_CHARS = 8_000
    task_md       = (Path(dataset_dir).parents[1] / "reference" / "task.md").read_text(encoding="utf-8")
    evaluate_path = os.path.join(dataset_dir, "evaluate.py")
    today         = date.today().isoformat()

    start_time = time.time()
    supervision_consecutive_fails = 0

    system_content = f"""\
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {today}

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'."""

    # ── Inner agent ────────────────────────────────────────────────────────────

    def _run_inner_agent(
        parent_node: dict | None,
        parent_code: str | None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Run a short multi-turn LLM that calls submit_solution to submit candidate solutions.
        The LLM may call submit_solution multiple times to iterate.
        Returns (submissions: list[dict], trajectory: list[dict]).
        """
        parent_score = float((parent_node.get("result") or {}).get("score", 0.0)) if parent_node else None

        if parent_code:
            parent_section = (
                f"**Parent solution** (score: {parent_score:.4f}) — produce a variation or improvement:\n\n"
                f"```python\n{parent_code[:3000]}\n```"
            )
        else:
            parent_section = (
                "**No parent — write from scratch.**\n\n"
                "Read the task description carefully — it describes the function signature, "
                "evaluation criteria, and optimization strategies to consider. "
                "Start with the simplest correct implementation, submit it to see the baseline score, "
                "then iterate."
            )

        developer_content = f"""\
# Instructions

{task_md}

## Your goal

Write a solution as described in the task above and submit it via `submit_solution(code)`.

{parent_section}

You may explore the dataset directory, read existing solutions in `{solutions_dir}/`, and run experiments.
When ready, call `submit_solution(code)` — it evaluates your code and returns the speedup score immediately.
If the score is low or there's an error, fix your code and call `submit_solution` again.
**Stop making tool calls only when you are satisfied with your score.**

Constraints:
- write_file: scratch files only (working_dir) — do NOT use it to submit solutions.
- read_file / bash: working_dir, solutions_dir (read-only), dataset_dir.

Working directory (scratch, read/write): {working_dir}
Solutions directory (read-only):         {solutions_dir}
Dataset directory (read-only):           {dataset_dir}

# Tools

## functions

{TOOLS_TS}"""

        messages = [
            {"role": "system",    "content": system_content},
            {"role": "developer", "content": developer_content},
        ]
        trajectory = []
        submissions: list[dict] = []

        for turn in range(1, max_inner_turns + 1):
            turns_left = max_inner_turns - turn
            if turn == 1:
                status = "Please produce and submit your solution."
            elif turns_left <= 1:
                status = (
                    f"[Turn {turn}/{max_inner_turns}] LAST TURN. "
                    "You MUST call submit_solution now if you have not submitted a working solution yet."
                )
            else:
                n_submitted = len(submissions)
                best_score  = max((s.get("score") or 0) for s in submissions) if submissions else None
                score_hint  = f" Best so far: {best_score:.4f}×." if best_score is not None else ""
                status = (
                    f"[Turn {turn}/{max_inner_turns}] {n_submitted} submission(s) so far.{score_hint} "
                    "Keep improving and call submit_solution when ready, or stop if satisfied."
                )
            messages.append({"role": "user", "content": status})

            try:
                response = call_task_model(
                    messages=messages,
                    model=args.model,
                    tools=TOOLS,
                    log_dir=log_dir,
                    temperature=args.task_model_temperature,
                )
            except Exception as _e:
                logger.warning(f"  Inner agent LLM error at turn {turn}: {_e}")
                trajectory.append({"turn": turn, "error": str(_e), "tool_calls": []})
                break

            tool_calls = response["tool_calls"]
            logger.info(f"  [inner t{turn}] tool_calls={[tc['name'] for tc in tool_calls]}")
            if tool_calls or not response["content"]:
                messages.extend(response["raw_messages"])
            else:
                messages.extend(
                    m for m in response["raw_messages"] if m.get("channel") in ("final", None)
                )

            turn_record = {"turn": turn, "tool_calls": [], "content": response.get("content", "")}

            if not tool_calls:
                trajectory.append(turn_record)
                break  # LLM stopped — done

            for tc in tool_calls:
                name, targs = tc["name"], tc["args"]
                if name == "bash":
                    out = _tools.bash(targs.get("command", ""), working_dir=working_dir, dataset_dir=dataset_dir, solutions_dir=solutions_dir)
                elif name == "read_file":
                    out = _tools.read_file(targs.get("path", ""), working_dir=working_dir, dataset_dir=dataset_dir, solutions_dir=solutions_dir)
                elif name == "write_file":
                    out = _tools.write_file(targs.get("path", ""), targs.get("content", ""), working_dir=working_dir)
                elif name == "submit_solution":
                    out = _tools.submit_solution(
                        targs.get("code", ""),
                        solutions_dir=solutions_dir,
                        working_dir=working_dir,
                        evaluate_path=evaluate_path,
                        state_file=args.state_file,
                        current_gen=args.current_gen,
                        write_node_fn=_write_tree_node_util,
                    )
                    try:
                        sub = json.loads(out)
                        submissions.append(sub)
                        logger.info(f"  → submit_solution: uid={sub.get('uid')} {sub.get('_status', '')}")
                    except Exception:
                        pass
                else:
                    out = f"[ERROR] unknown tool: {name}"
                if len(out) > MAX_OUTPUT_CHARS:
                    out = out[:MAX_OUTPUT_CHARS] + f"\n... [truncated — {len(out)} total chars]"
                messages.append({"role": "tool_result", "name": name, "content": out, "call_id": tc.get("call_id")})
                turn_record["tool_calls"].append({
                    "name": name,
                    "args": {k: v[:200] if isinstance(v, str) and len(v) > 200 else v for k, v in targs.items()},
                    "result_preview": out[:300],
                })

            trajectory.append(turn_record)

        return submissions, trajectory

    # ── Seed tree with reference solution (if provided) ───────────────────────

    if args.seed_solution and os.path.exists(args.seed_solution) and args.state_file:
        from utils.tree_utils import get_generation_nodes
        if not get_generation_nodes(args.state_file, args.current_gen):
            import shutil as _shutil
            seed_uid  = str(_uuid_mod.uuid4())[:8]
            seed_dest = os.path.join(solutions_dir, f"{seed_uid}.py")
            _shutil.copy2(args.seed_solution, seed_dest)
            logger.info(f"[seed] Evaluating reference solution → {seed_dest}")
            try:
                proc = subprocess.run(
                    [sys.executable, evaluate_path, seed_dest],
                    capture_output=True, text=True, cwd=working_dir, timeout=300,
                )
                full_out = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
                if "RESULT_JSON:" in full_out:
                    json_str    = full_out.split("RESULT_JSON:", 1)[1].split("\n", 1)[0].strip()
                    seed_result = json.loads(json_str)
                    if isinstance(seed_result, dict) and "score" in seed_result and not seed_result.get("error"):
                        node_id = _write_tree_node_util(args.state_file, seed_dest, seed_result, args.current_gen)
                        logger.info(f"[seed] Node written: {node_id}  score={seed_result['score']:.4f}")
                    else:
                        logger.warning(f"[seed] Evaluation error: {seed_result.get('error')}")
                else:
                    logger.warning(f"[seed] No RESULT_JSON in output — seed skipped")
            except Exception as _e:
                logger.warning(f"[seed] Failed: {_e}")
        else:
            logger.info("[seed] Tree already has nodes — skipping seed")

    # ── Outer MCTS loop ────────────────────────────────────────────────────────

    outer_turn = 0
    mcts_log: list[dict] = []

    while True:
        outer_turn += 1
        elapsed = time.time() - start_time
        logger.info(f"MCTS iteration {outer_turn} — {elapsed:.0f}s elapsed")

        # 1. PUCT: select parent node
        parent_node, parent_code = (
            _select_parent(args.state_file, c_puct) if args.state_file
            else (None, None)
        )
        parent_score = float((parent_node.get("result") or {}).get("score", 0.0)) if parent_node else None

        iter_record: dict = {
            "iteration":        outer_turn,
            "elapsed_s":        round(elapsed, 1),
            "parent_score":     parent_score,
            "submissions":      [],
            "best_score":       None,
            "inner_trajectory": [],
        }

        # 3. Run inner agent — submit_solution handles eval + state.json registration
        submissions, inner_traj = _run_inner_agent(parent_node, parent_code)

        if not submissions:
            logger.warning(f"  Iter {outer_turn}: no submit_solution called")
        else:
            best = max(submissions, key=lambda s: float(s.get("score") or 0))
            logger.info(f"  Iter {outer_turn}: {len(submissions)} submission(s), best={best.get('score')}")

        iter_record["inner_trajectory"] = inner_traj
        iter_record["submissions"]      = submissions
        iter_record["best_score"]       = max((s.get("score") or 0 for s in submissions), default=None) if submissions else None
        mcts_log.append(iter_record)
        Path(agent_execution_path).write_text(
            json.dumps(mcts_log, indent=2, default=str), encoding="utf-8"
        )

        # 6. Supervision
        if args.supervision_model and outer_turn % args.supervision_interval == 0:
            elapsed = time.time() - start_time
            remaining = max(0, args.exp_duration_seconds - elapsed) if args.exp_duration_seconds > 0 else 0
            try:
                summary = summarize_tree(args.state_file)
            except Exception as _e:
                Path(exit_reason_path).write_text(f"broken_gen: state.json tree summary error — {_e}")
                break
            sup = check_supervision(
                tree_summary=summary,
                elapsed_seconds=elapsed,
                remaining_seconds=remaining,
                current_gen=args.current_gen,
                model=args.supervision_model,
                gen0_evolve_duration_s=args.gen0_evolve_duration,
            )
            decision, sup_reason = sup["decision"], sup["reason"]
            if decision == "fail":
                supervision_consecutive_fails += 1
                with open(supervision_log_path, "a") as _lf:
                    _lf.write(f"{datetime.datetime.now().isoformat()} iter={outer_turn} FAIL consecutive={supervision_consecutive_fails}: {sup_reason}\n")
                logger.warning(f"  Supervision FAIL (consecutive {supervision_consecutive_fails}): {sup_reason}")
                if supervision_consecutive_fails >= 2:
                    Path(exit_reason_path).write_text(f"broken_gen: 2 consecutive supervision failures — {sup_reason}")
                    break
            else:
                supervision_consecutive_fails = 0
                with open(supervision_log_path, "a") as _lf:
                    _lf.write(f"{datetime.datetime.now().isoformat()} iter={outer_turn} {decision.upper()}: {sup_reason}\n")
                logger.info(f"  Supervision {decision.upper()} — {sup_reason}")
                if decision in ("evolve", "stop"):
                    Path(exit_reason_path).write_text(f"{decision}: {sup_reason}")
                    break


if __name__ == "__main__":
    try:
        main()
    except Exception as _top_e:
        import traceback as _tb
        _msg = f"broken_gen: unhandled exception — {_top_e}"
        logger.error(f"{_msg}\n{_tb.format_exc()}")
        try:
            _erp = sys.argv[sys.argv.index("--exit_reason_path") + 1]
            Path(_erp).write_text(_msg)
        except Exception:
            pass
        sys.exit(1)
