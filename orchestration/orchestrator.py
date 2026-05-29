"""
Directory structure

orchestration/
  orchestrator.py

tasks/
  task_1/
    reference/
      reference_target_agent/   ← target-agent package template
        target_agent.py
        conf.yaml
      task.md                   ← unified task description (meta-agent + target agent)
    data/
      public/
      private/

runs/
  run_1/
    context.md
    research_state/             ← persists across ALL generations
      tree_state.json
      solutions/
      logs/
      artifacts/
    meta_agent_memory/
      generation_001/           ← prompt saved for each meta-agent call
      generation_002/
    gen_1/
      target_agent/             ← package (directory)
        target_agent.py
        conf.yaml
      target_agent_stdout.log
      agent_execution.json
      results.json
      improvement.md            ← written by meta-agent for gen 2+
      meta_agent_prompt.txt
      solution.py
    gen_2/
      ...

Generation flow:
  gen_1: meta-agent (reference agent, empty tree) → target_agent/ package
  gen_2: meta-agent (gen_1 execution + tree state)  → target_agent/ package
  gen_N: meta-agent (gen_{N-1} execution + tree state) → target_agent/ package
"""

import os
import re
import sys
import json
import shutil
import signal
import asyncio
import logging
import argparse
import glob
import time
from pathlib import Path
from datetime import datetime

from util import run_agent
from model_guidelines import get_guidelines
from plot_utils import plot_scores as _plot_scores

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    return open(os.path.join(_PROMPTS_DIR, filename), encoding="utf-8").read()


def _fill_template(template: str, **kwargs) -> str:
    """Replace {UPPER_CASE_KEY} placeholders in template. Single-pass — safe against cascading."""
    def _replace(m):
        return str(kwargs.get(m.group(1), m.group(0)))
    return re.sub(r"\{([A-Z_][A-Z0-9_]*)\}", _replace, template)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ========================
# HELPER FUNCTIONS
# ========================

_current_proc: "subprocess.Popen | None" = None


def _kill_proc(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError, ChildProcessError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=2)
        except (ProcessLookupError, ChildProcessError, OSError):
            pass


def _kill_current_proc() -> None:
    global _current_proc
    if _current_proc is None:
        return
    _kill_proc(_current_proc)
    _current_proc = None


def _signal_handler(signum, frame):
    logger.warning(f"Received signal {signum} — killing subprocess and exiting.")
    _kill_current_proc()
    sys.exit(1)


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _run_command(command: str, cwd: str | None = None) -> int:
    """Run a shell command, waiting until completion. Returns the return code."""
    global _current_proc
    proc = subprocess.Popen(
        command, shell=True, executable="/bin/bash", text=True,
        start_new_session=True, cwd=cwd,
    )
    _current_proc = proc
    proc.wait()
    _current_proc = None
    return proc.returncode


def _build_rich_tree_summary(gen_dir: str, run_dir: str) -> tuple:
    """Returns (summary_text, is_valid). is_valid=False triggers broken_gen."""
    tree_path = os.path.join(gen_dir, "state.json")
    if not os.path.exists(tree_path):
        return "No state.json found.", True
    try:
        with open(tree_path) as f:
            tree = json.load(f)
    except Exception as e:
        return f"Could not parse state.json: {e}", False

    nodes = tree.get("nodes", {})
    root_id = tree.get("root_id")
    evaluated = [(nid, n) for nid, n in nodes.items()
                 if n.get("result") is not None and nid != root_id]

    if not evaluated:
        return (f"Tree state: {len(nodes)} nodes total, 0 evaluated.\n"
                f"Run directory: {run_dir}"), True

    # Validate required keys
    errors = []
    for nid, n in evaluated:
        if "generation" not in n:
            errors.append(f"node {nid}: missing 'generation'")
        if "uuid" not in n:
            errors.append(f"node {nid}: missing 'uuid'")
        result = n.get("result")
        if not isinstance(result, dict) or "score" not in result:
            errors.append(f"node {nid}: 'result' must be a dict with 'score' key")
    if errors:
        return "VALIDATION ERRORS (broken_gen):\n" + "\n".join(errors), False

    evaluated.sort(key=lambda x: x[1]["result"]["score"], reverse=True)
    best_nid, best_node = evaluated[0]
    best_score = best_node["result"]["score"]
    best_gen = best_node.get("generation", "?")

    gen_bests: dict = {}
    for nid, n in evaluated:
        g = n.get("generation", 0)
        s = n["result"]["score"]
        if g not in gen_bests or s > gen_bests[g][1]:
            gen_bests[g] = (nid, s)

    recent_gens = sorted(gen_bests.keys())[-6:]
    recent_scores = [gen_bests[g][1] for g in recent_gens]
    stagnant = (len(recent_scores) >= 3 and
                all(abs(recent_scores[i] - recent_scores[i-1]) < 0.001
                    for i in range(1, len(recent_scores))))
    improving = len(recent_scores) >= 2 and recent_scores[-1] > recent_scores[0]
    trend = " -> ".join(f"gen{g}:{gen_bests[g][1]:.4f}" for g in recent_gens)
    status = ("STAGNATING" if stagnant else "Improving" if improving else "Exploring")

    top_lines = []
    for nid, n in evaluated[:5]:
        extra = {k: v for k, v in n["result"].items() if k != "score"}
        extra_str = ("  " + json.dumps(extra)[:100]) if extra else ""
        top_lines.append(
            f"  {nid}  gen={n.get('generation','?')}  score={n['result']['score']:.4f}"
            f"  uuid={str(n.get('uuid','?'))[:8]}  path={n.get('solution_path','N/A')}{extra_str}"
        )

    return (
        f"=== Tree State Summary ===\n"
        f"Nodes: {len(nodes)} total, {len(evaluated)} evaluated\n"
        f"Best:  score={best_score:.4f}  node={best_nid}  gen={best_gen}\n"
        f"       path={best_node.get('solution_path', 'N/A')}\n"
        f"Trend: {trend}\n"
        f"Status: {status}\n\nTop nodes:\n" + "\n".join(top_lines) +
        f"\n\nRun directory: {run_dir}\n"
        f"State file:    {tree_path}\n"
        f"best_node_id:  {tree.get('best_node_id')}\n"
    ), True


def _read_tree_summary(gen_dir: str) -> str:
    text, _ = _build_rich_tree_summary(gen_dir, RUN_DIRECTORY)
    return text


def _init_gen_state(gen_dir: str, task_id: str) -> None:
    """Create an empty state.json in gen_dir if one doesn't exist."""
    state_path = os.path.join(gen_dir, "state.json")
    if not os.path.exists(state_path):
        initial = {
            "version": "0.2", "task_id": task_id,
            "root_id": "node_0000", "best_node_id": None,
            "nodes": {"node_0000": {
                "id": "node_0000", "parent": None, "children": [],
                "uuid": "root", "generation": 0,
                "solution_path": None, "result": None,
                "visits": 0, "status": "root", "metadata": {},
            }},
        }
        with open(state_path, "w") as f:
            json.dump(initial, f, indent=2)
        logger.info(f"  ✓ Initialized state.json in {os.path.basename(gen_dir)}")
    else:
        tree = json.load(open(state_path))
        logger.info(f"  ✓ Existing state.json: {len(tree.get('nodes', {}))} nodes")


def _copy_gen_forward(src_dir: str, dst_dir: str) -> None:
    """Copy only the persistent scaffold state to the next generation directory.

    Whitelist — only these items are carried forward:
      target_agent/   — the scaffold the meta-agent will modify
      state.json      — the persistent search tree
      solutions/      — evaluated solutions (pruned to referenced only)
      state_summary.md — tree summary read by the meta-agent
      context.md      — run history read by the meta-agent

    Everything else (logs, execution traces, exit_reason.txt, etc.) is
    generation-specific and must be written fresh by each new generation.
    """
    _WHITELIST = {"target_agent", "state.json", "solutions", "state_summary.md", "context.md"}

    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name not in _WHITELIST:
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Prune solutions/ — keep only files referenced by a node in state.json
    solutions_dst = os.path.join(dst_dir, "solutions")
    state_path    = os.path.join(dst_dir, "state.json")
    if os.path.isdir(solutions_dst) and os.path.exists(state_path):
        try:
            with open(state_path) as f:
                tree = json.load(f)
            referenced = {
                os.path.basename(n["solution_path"])
                for n in tree.get("nodes", {}).values()
                if n.get("solution_path")
            }
            removed = 0
            for fname in os.listdir(solutions_dst):
                if fname not in referenced:
                    fpath = os.path.join(solutions_dst, fname)
                    if os.path.isdir(fpath):
                        shutil.rmtree(fpath)
                    else:
                        os.remove(fpath)
                    removed += 1
            if removed:
                logger.info(f"  ✓ Pruned {removed} unreferenced solution(s) from solutions/")
        except Exception as e:
            logger.warning(f"  ⚠ Could not prune solutions/: {e}")

    logger.info(f"  ✓ Copied {os.path.basename(src_dir)} → {os.path.basename(dst_dir)}")


def _fallback_gen(current_gen: int, run_dir: str) -> bool:
    """Move broken gen dir to gen_broken/, re-copy from prev gen. Returns True if successful."""
    current_dir = os.path.abspath(os.path.join(run_dir, f"gen_{current_gen}"))
    prev_dir    = os.path.abspath(os.path.join(run_dir, f"gen_{current_gen - 1}"))
    if not os.path.exists(prev_dir):
        logger.warning(f"  ⚠ No gen_{current_gen - 1} to fall back to")
        return False
    broken_base = os.path.join(run_dir, "gen_broken")
    os.makedirs(broken_base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    broken_dest = os.path.join(broken_base, f"gen_{current_gen}_{ts}")
    shutil.move(current_dir, broken_dest)
    logger.info(f"  → Moved gen_{current_gen} → gen_broken/gen_{current_gen}_{ts}")
    _copy_gen_forward(prev_dir, current_dir)
    logger.info(f"  ✓ Fallback: fresh copy of gen_{current_gen - 1} → gen_{current_gen}")
    return True


def _write_state_summary(gen_dir: str) -> str:
    """Read state.json, write state_summary.md, return summary text."""
    text, is_valid = _build_rich_tree_summary(gen_dir, RUN_DIRECTORY)
    Path(os.path.join(gen_dir, "state_summary.md")).write_text(text)
    return text


def _llm_context_summary(
    gen_num: int,
    best_score: float | None,
    prev_best_score: float | None,
    exit_reason: str,
    improvement_md: str,
    meta_model: str,
) -> str:
    """Call the supervision model for a 2-3 sentence gen summary. Returns "" on failure."""
    try:
        import litellm
        score_line = ""
        if best_score is not None and best_score > 0:
            score_line = f"Best score this gen: {best_score:.4f}"
            if prev_best_score is not None and prev_best_score > 0:
                delta = best_score - prev_best_score
                score_line += f" (Δ {delta:+.4f} vs previous gen)"
        prompt = (
            f"Summarize generation {gen_num} of a self-improving search agent in 2-3 sentences.\n\n"
            f"{score_line}\n"
            f"Exit reason: {exit_reason or 'natural completion'}\n\n"
            f"Meta-agent improvement plan (excerpt):\n{improvement_md[:600]}\n\n"
            "Focus on: what strategy was tried, how scores changed, and what to watch next."
        )
        resp = litellm.completion(
            model=meta_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as _e:
        logger.debug(f"Context LLM summary skipped: {_e}")
        return ""


def _append_context(
    gen_dir: str,
    gen_num: int,
    tree_summary: str,
    success: bool,
    duration: float,
    best_score: float | None = None,
    prev_best_score: float | None = None,
    exit_reason: str = "",
    improvement_md_path: str | None = None,
    meta_model: str | None = None,
) -> None:
    """Append a generation entry to context.md in gen_dir (cumulative across gens)."""
    ctx_path = os.path.join(gen_dir, "context.md")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mark = "✓" if success else "✗"

    lines = [f"\n\n## Gen {gen_num} — {ts} [{mark}] ({duration:.0f}s)"]

    if best_score is not None and best_score > 0:
        score_str = f"**Best score**: {best_score:.4f}"
        if prev_best_score is not None and prev_best_score > 0:
            delta = best_score - prev_best_score
            score_str += f"  (Δ {delta:+.4f})"
        lines.append(score_str)

    if exit_reason:
        lines.append(f"**Exit**: `{exit_reason}`")

    # Optional LLM summary
    improvement_md = ""
    if improvement_md_path and os.path.exists(improvement_md_path):
        try:
            improvement_md = Path(improvement_md_path).read_text(encoding="utf-8")
        except Exception:
            pass

    if meta_model and (improvement_md or best_score):
        summary = _llm_context_summary(
            gen_num, best_score, prev_best_score, exit_reason, improvement_md, meta_model
        )
        if summary:
            lines.append(f"\n**Summary**: {summary}")

    lines.append(f"\n**Search state**:\n{tree_summary[:800]}")

    entry = "\n".join(lines)
    if os.path.exists(ctx_path):
        Path(ctx_path).write_text(Path(ctx_path).read_text() + entry)
    else:
        Path(ctx_path).write_text(f"# Run history\n{entry}")


def load_agent_execution(gen_directory):
    """Load execution logs. Supports single-file and multi-trajectory formats."""
    execution_folder = os.path.join(gen_directory, "agent_execution")
    execution_file   = os.path.join(gen_directory, "agent_execution.json")

    if os.path.isdir(execution_folder):
        logger.info(f"  → Detected multi-trajectory format (folder)")
        files = sorted(glob.glob(os.path.join(execution_folder, "execution_q*.json")))
        if not files:
            return {"error": "Empty execution folder", "type": "multi-trajectory"}, True
        trajectories = []
        for fp in files:
            try:
                with open(fp) as fh:
                    trajectories.append(json.load(fh))
            except Exception as e:
                trajectories.append({"error": str(e), "file": os.path.basename(fp)})
        logger.info(f"  ✓ Loaded {len(trajectories)} trajectory files")
        return {"trajectories": trajectories, "count": len(trajectories), "type": "multi-trajectory"}, True

    elif os.path.exists(execution_file):
        try:
            with open(execution_file) as f:
                data = json.load(f)
            return data, False
        except json.JSONDecodeError as e:
            try:
                raw = Path(execution_file).read_text()
                return {"error": "Parse error", "raw_preview": raw[:1000], "parse_error": str(e)}, False
            except Exception as re:
                return {"error": "Could not read file", "read_error": str(re)}, False
        except FileNotFoundError:
            return {"error": "Execution log file not found"}, False
    else:
        return {"error": "Execution log not found"}, False


# ========================
# ARGUMENT PARSING
# ========================

parser = argparse.ArgumentParser(description='SIA orchestrator — persistent tree-search agent evolution')
parser.add_argument('--run_id', type=int, default=1, help='Run ID (default: 1)')
parser.add_argument('--task_dir', type=str, required=True, help='Path to the task directory')
parser.add_argument('--meta_model', type=str, default=None, help='Model for the meta-agent (all generations)')
parser.add_argument('--task_model', type=str, default='claude-haiku-4-5-20251001', help='Model for the target agent')
parser.add_argument('--backend', type=str, default='claude', choices=['claude', 'openhands'])
parser.add_argument('--task_model_temperature', type=float, default=0.3)
parser.add_argument('--exp_duration_min', type=int, default=60,
                    help='Total experiment budget in minutes; safety timeout = 1.3× this (default: 60)')
# gen_0 is always the reference agent — no flag needed
parser.add_argument('--supervision_model', type=str, default='gemini/gemini-3.1-pro-preview',
                    help='Model for per-iteration supervision calls inside the target agent')
parser.add_argument('--supervision_interval', type=int, default=5,
                    help='Run supervision every N turns inside the target agent (default: 5)')
parser.add_argument('--supervision_check_timeout_min', type=int, default=5,
                    help='Minutes without a supervision heartbeat before broken_gen (default: 5)')
parser.add_argument('--gen0_evolve_duration', type=int, default=900,
                    help='Gen-0: EVOLVE after this many seconds without LLM call (default: 900 = 15 min)')
parser.add_argument('--meta_agent_max_turns', type=int, default=100,
                    help='Max turns for the meta-agent tool loop (default: 100)')
parser.add_argument('--seed_solution', type=str, default=None,
                    help='Path to a reference solution (.py) to evaluate and seed the gen-0 tree before the MCTS loop')
args = parser.parse_args()

task_dir   = args.task_dir
run_id     = args.run_id
backend    = args.backend

if args.meta_model is None:
    meta_model = 'gemini/gemini-3.1-pro-preview' if backend == 'openhands' else 'haiku'
    logger.info(f"Using default model for {backend} backend: {meta_model}")
else:
    meta_model = args.meta_model

task_model = args.task_model


def _required_api_key(model: str) -> tuple[str, list[str]]:
    m = model.lower()
    if any(m.startswith(p) for p in ("gemini/", "google/")):
        return "Gemini / Google", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    if "gpt-oss" in m or "tinker" in m:
        return "Tinker", ["TINKER_API_KEY"]
    if any(m.startswith(p) for p in ("claude", "anthropic/")):
        return "Anthropic", ["ANTHROPIC_API_KEY"]
    return "OpenAI", ["OPENAI_API_KEY"]


def _check_api_key(model: str) -> None:
    provider, candidates = _required_api_key(model)
    if all(not os.environ.get(k) for k in candidates):
        logger.error(f"Model '{model}' requires {provider} credentials — set {' or '.join(candidates)}")
        sys.exit(1)


_check_api_key(meta_model)
_check_api_key(task_model)

logger.info(f"Configuration:")
logger.info(f"  - Budget:         {args.exp_duration_min} min (safety timeout: {args.exp_duration_min * 1.3:.0f} min)")
logger.info(f"  - Task:           {task_dir}")
logger.info(f"  - Run ID:         {run_id}")
logger.info(f"  - Backend:        {backend}")
logger.info(f"  - Meta-agent:     {meta_model}")
logger.info(f"  - Target-agent:   {task_model}")
logger.info(f"  - Supervision:    {args.supervision_model} every {args.supervision_interval} turns")


# ========================
# SECTION 1: Load Files
# ========================

logger.info("Loading task files...")

_ref_pkg  = os.path.join(task_dir, "reference/reference_target_agent")
_ref_file = os.path.join(task_dir, "reference/reference_target_agent.py")
if os.path.isdir(_ref_pkg):
    REFERENCE_TARGET_AGENT_PY = open(os.path.join(_ref_pkg, "target_agent.py")).read()
else:
    REFERENCE_TARGET_AGENT_PY = open(_ref_file).read()
logger.info("  ✓ Reference target agent loaded")

TASK_MD                = open(os.path.join(task_dir, "reference/task.md")).read()
SCAFFOLD_DESIGN_GUIDELINES = _load_prompt("scaffold_design_guidelines.md")
TARGET_AGENT_SPEC          = _load_prompt("target_agent_spec.md")
_META_AGENT_TEMPLATE       = _load_prompt("meta_agent_prompt.md")
logger.info("  ✓ All task files and prompt templates loaded")


# ========================
# SECTION 2: Setup Run Directories
# ========================

gen_num = 0  # always start from gen_0 (reference agent)
RUN_DIRECTORY = f"./runs/run_{run_id}"

if os.path.exists(RUN_DIRECTORY):
    logger.error(f"Run directory already exists: {RUN_DIRECTORY}. Use a different --run_id.")
    sys.exit(1)

os.makedirs(RUN_DIRECTORY, exist_ok=False)
first_gen_directory = os.path.abspath(f"{RUN_DIRECTORY}/gen_{gen_num}")
os.makedirs(first_gen_directory, exist_ok=False)

task_id = os.path.basename(os.path.abspath(task_dir))
_init_gen_state(first_gen_directory, task_id)

import venv
import subprocess

venv_dir = os.path.join(RUN_DIRECTORY, "venv")
logger.info(f"Creating venv at: {venv_dir}")
uv_available = subprocess.run(["which", "uv"], capture_output=True).returncode == 0
if uv_available:
    subprocess.run(["uv", "venv", "--python", "3.12", venv_dir], check=True)
else:
    venv.create(venv_dir, with_pip=True)

pip_executable = os.path.join(venv_dir, "bin", "pip")

def pip_install(pip_args):
    if uv_available:
        subprocess.run(
            ["uv", "pip", "install", "--python", pip_executable.replace("/bin/pip", "/bin/python")] + pip_args,
            check=True
        )
    else:
        subprocess.run([pip_executable, "install"] + pip_args, check=True)

pip_install(["-r", os.path.abspath(os.path.join(task_dir, "../_shared/base_requirements.txt"))])

task_requirements = os.path.join(task_dir, "requirements.txt")
if os.path.exists(task_requirements):
    pip_install(["-r", task_requirements])
    logger.info("  ✓ Task requirements installed")

pip_install(["pyyaml"])

Path(first_gen_directory, "context.md").write_text(
    f"# Run history\nTask: {task_id}  Meta: {meta_model}  Agent: {task_model}\n"
)


# ========================
# SECTION 3: Meta-Agent Prompt Builder
# ========================

TASK_MODEL_GUIDELINES = get_guidelines(task_model)
_guidelines_section   = f"\n---\n{TASK_MODEL_GUIDELINES}\n---\n" if TASK_MODEL_GUIDELINES else ""
DATASET_DIRECTORY     = os.path.join(task_dir, "data/public")
ABS_DATASET_DIRECTORY = os.path.abspath(DATASET_DIRECTORY)
ABS_SHARED_DIRECTORY  = os.path.abspath(os.path.join(task_dir, "../_shared"))


def _get_broken_gen_history(run_dir: str, gen_num: int) -> str:
    """Scan gen_broken/ for past failed attempts at gen_num and return a warning block.

    Looks for gen_broken/gen_{gen_num}_<timestamp>/ directories, reads their
    exit_reason.txt and improvement.md, and formats them as a prompt section.
    Returns "" if no prior broken attempts exist.
    """
    broken_base = os.path.join(run_dir, "gen_broken")
    if not os.path.isdir(broken_base):
        return ""
    prefix = f"gen_{gen_num}_"
    attempts = sorted(
        d for d in os.listdir(broken_base)
        if d.startswith(prefix) and os.path.isdir(os.path.join(broken_base, d))
    )
    if not attempts:
        return ""

    lines = [
        f"## ⚠ Prior failed attempts at generation {gen_num} — do NOT repeat these mistakes",
        "",
        f"Generation {gen_num} has already been attempted {len(attempts)} time(s) and failed.",
        "Study each failure below and make sure your scaffold avoids the same issues.",
        "",
    ]
    for attempt_dir in attempts:
        full_path = os.path.join(broken_base, attempt_dir)
        exit_reason = ""
        improvement = ""
        try:
            er_path = os.path.join(full_path, "exit_reason.txt")
            if os.path.exists(er_path):
                exit_reason = Path(er_path).read_text(encoding="utf-8").strip()
        except Exception:
            pass
        try:
            imp_path = os.path.join(full_path, "improvement.md")
            if os.path.exists(imp_path):
                improvement = Path(imp_path).read_text(encoding="utf-8").strip()[:600]
        except Exception:
            pass
        lines.append(f"### {attempt_dir}")
        if exit_reason:
            lines.append(f"**Exit reason**: `{exit_reason}`")
        if improvement:
            lines += ["**Improvement plan that was tried** (excerpt):", f"```\n{improvement}\n```"]
        if not exit_reason and not improvement:
            lines.append("_(no exit_reason or improvement.md found)_")
        lines.append("")

    return "\n".join(lines)


def _build_meta_agent_prompt(
    gen_num: int,
    improvement_dir: str,
    *,
    # populated for gen 2+
    agent_py: str = "",
    conf_yaml: str = "",
    best_agent_section: str = "",
    best_score: float = -1.0,
    best_gen: int = 0,
    tree_state_summary: str = "",
    exp_duration_min: int = 60,
    fallback_warning: str = "",
    prior_broken_history: str = "",
) -> str:
    """Build the meta-agent prompt for any generation.

    Gen 0: reference agent, no meta-agent call.
    Gen 1+: meta-agent modifies existing files (working dir is copy of previous gen).
    Gen 1 is treated identically to gen 2+ — it always has gen_0 data available.
    """
    # ── Generation note ──────────────────────────────────────────────────────
    generation_note = (
        f"**This is generation {gen_num}.**\n"
        f"Your goal: produce a scaffold that scores **higher than {best_score:.4f}** "
        f"(the best score so far, from generation {best_gen})."
    )

    # ── Generation context section ────────────────────────────────────────────
    generation_context = f"""\
- Current generation: {gen_num}
- Best score so far: **{best_score:.4f}** (Generation {best_gen})
- Working directory (gen_{gen_num} — already populated from gen_{gen_num-1}): {improvement_dir}

**State summary** (from gen_{gen_num-1}, your primary context):
{tree_state_summary}

Read `context.md` and `state_summary.md` in your working directory for full history.
"""

    # ── Agent starting point ──────────────────────────────────────────────────
    agent_starting_point = (
        f"**Your working directory `{improvement_dir}` is a copy of gen_{gen_num-1}.**\n"
        f"The files already exist — **do NOT create them from scratch**.\n"
        f"**MODIFY** `target_agent/target_agent.py` (and optionally `target_agent/conf.yaml`) "
        f"to improve performance beyond {best_score:.4f}.\n\n"
        f"{best_agent_section}"
    )

    # ── improvement.md note ───────────────────────────────────────────────────
    improvement_md_note = (
            "Create **`improvement.md`** in your working directory documenting "
            "what you changed and why."
        )

    # ── Sample task descriptions ──────────────────────────────────────────────
    if fallback_warning:
        generation_context = fallback_warning + "\n\n---\n\n" + generation_context
    if prior_broken_history:
        generation_context = prior_broken_history + "\n\n---\n\n" + generation_context

    readme_path = os.path.join(improvement_dir, "target_agent", "README.md")
    try:
        target_agent_readme = Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        target_agent_readme = "*(no README.md found in target_agent/)*"

    prompt = _fill_template(
        _META_AGENT_TEMPLATE,
        GENERATION_NOTE=generation_note,
        GENERATION_CONTEXT=generation_context,
        TASK_MD=TASK_MD,
        AGENT_STARTING_POINT=agent_starting_point,
        TARGET_AGENT_README=target_agent_readme,
        TARGET_AGENT_SPEC=TARGET_AGENT_SPEC,
        IMPROVEMENT_DIR=improvement_dir,
        IMPROVEMENT_MD_NOTE=improvement_md_note,
        TASK_MODEL=task_model,
        REQUIRED_API_KEYS=str(_required_api_key(task_model)[1]),
        EXP_DURATION_MIN=str(exp_duration_min),
        SCAFFOLD_DESIGN_GUIDELINES=SCAFFOLD_DESIGN_GUIDELINES,
        TASK_MODEL_GUIDELINES_SECTION=_guidelines_section,
        VENV_PIP=pip_executable,
    )
    return prompt


def _run_meta_agent(prompt: str, working_dir: str, gen_num: int) -> bool:
    """Run the meta-agent and return True if target_agent/target_agent.py was created."""
    prompt_path = os.path.join(working_dir, "meta_agent_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(prompt)

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            logger.warning(f"  ↻ Meta-agent retry {attempt}/{MAX_RETRIES} — target_agent/target_agent.py not found")
        asyncio.run(run_agent(
            model_name=meta_model, max_turns=str(args.meta_agent_max_turns), prompt=prompt,
            agent_working_directory=working_dir, backend=backend,
        ))
        if Path(working_dir, "target_agent", "target_agent.py").exists():
            logger.info(f"  ✓ target_agent/target_agent.py created (attempt {attempt})")
            return True
        logger.warning(f"  ✗ Not found after attempt {attempt}/{MAX_RETRIES}")
    logger.error(f"  ✗ Meta-agent failed after {MAX_RETRIES} attempts")
    return False


# ========================
# SECTION 4: Generation 0 — Bootstrap from reference
# ========================

# gen_0 is always the reference agent, copied verbatim (no meta-agent for gen_0)
dest_pkg = os.path.join(first_gen_directory, "target_agent")
if os.path.isdir(_ref_pkg):
    shutil.copytree(_ref_pkg, dest_pkg)
    logger.info(f"  ✓ Gen 0: copied reference_target_agent/ → gen_0/target_agent/")
else:
    os.makedirs(dest_pkg, exist_ok=True)
    shutil.copy(_ref_file, os.path.join(dest_pkg, "target_agent.py"))
    Path(dest_pkg, "conf.yaml").write_text(
        "search:\n"
        "  n_candidates: 3\n"
        "  c_puct: 1.5\n"
        "  branching_factor: 2\n"
        "  max_depth: 5\n\n"
        "exploration:\n"
        "  temperature: 1.0\n"
        "  top_k_parents: 3\n"
        "  restart_on_stagnation: true\n"
        "  stagnation_patience: 5\n"
    )
    logger.info(f"  ✓ Gen 0: wrapped reference_target_agent.py → gen_0/target_agent/")


def _build_fallback_warning(
    gen_num: int,
    error_msg: str,
    exit_reason: str,
    state_issue: str,
    broken_agent_py: str,
    stdout_tail: str,
) -> str:
    """Build a prominent warning block for the meta-agent when a broken_gen occurred."""
    lines = [
        f"## ⚠ WARNING — Generation {gen_num} previously failed (broken_gen)",
        "",
        "The previous attempt at this generation was aborted. "
        "Study what broke and **do not replicate it**.",
        "",
    ]
    if error_msg:
        lines += [f"**Orchestrator error**: `{error_msg}`"]
    if exit_reason:
        lines += [f"**Exit reason**: `{exit_reason}`"]
    if state_issue:
        lines += [f"**State issue**: {state_issue}"]
    if stdout_tail:
        tail = stdout_tail[-2000:].strip()
        lines += ["", "**Last output from the failed agent** (tail):", "```", tail, "```"]
    if broken_agent_py:
        preview = broken_agent_py[:4000]
        truncated = " [truncated]" if len(broken_agent_py) > 4000 else ""
        lines += [
            "",
            f"**Broken `target_agent.py`**{truncated} — the agent that failed:",
            "```python",
            preview,
            "```",
        ]
    lines += [
        "",
        "**Action**: identify the root cause above, fix it in your new scaffold, "
        "and verify the fix before writing `target_agent.py`.",
    ]
    return "\n".join(lines)


# ========================
# SECTION 5: Safety Timer + Main Loop
# ========================

import threading as _threading

def _run_final_private_scores() -> None:
    """Find the best solution across all gens and run private scoring. Called by safety timer."""
    private_eval = os.path.join(task_dir, "data/private/evaluate.py")
    if not os.path.exists(private_eval):
        return
    best_sol, best_score_fs, best_gen_fs = None, -1.0, -1
    gen_num = 0
    while True:
        gen_dir = os.path.abspath(f"{RUN_DIRECTORY}/gen_{gen_num}")
        if not os.path.isdir(gen_dir):
            break
        try:
            with open(os.path.join(gen_dir, "state.json")) as _f:
                tree = json.load(_f)
            for _n in tree.get("nodes", {}).values():
                _r = _n.get("result")
                if isinstance(_r, dict) and "score" in _r and _n.get("solution_path"):
                    _s = float(_r["score"])
                    if _s > best_score_fs:
                        best_score_fs = _s
                        best_sol = _n["solution_path"]
                        best_gen_fs = gen_num
        except Exception:
            pass
        gen_num += 1
    if not best_sol or not os.path.exists(best_sol):
        logger.warning("[safety] No valid solution found for final private scoring")
        return
    logger.info(f"[safety] Running final private score — gen_{best_gen_fs} public={best_score_fs:.4f}")
    priv_work_dir = os.path.abspath(
        os.path.join(RUN_DIRECTORY, "private_scores", f"final_gen{best_gen_fs}")
    )
    os.makedirs(priv_work_dir, exist_ok=True)
    subprocess.run([python_exec, private_eval, best_sol], cwd=priv_work_dir)


_EXP_START = time.time()
_SAFETY_TIMEOUT_S = args.exp_duration_min * 60 * 1.3

def _safety_timer_fn():
    time.sleep(_SAFETY_TIMEOUT_S)
    logger.warning(
        f"⏰ Safety timeout reached ({_SAFETY_TIMEOUT_S:.0f}s = "
        f"{args.exp_duration_min * 1.3:.0f} min) — killing agent and finalizing."
    )
    _kill_current_proc()
    _run_final_private_scores()
    logging.shutdown()
    os._exit(0)

_safety_thread = _threading.Thread(target=_safety_timer_fn, daemon=True, name="safety-timer")
_safety_thread.start()
logger.info(f"Safety timer started: {_SAFETY_TIMEOUT_S:.0f}s ({args.exp_duration_min * 1.3:.0f} min)")


def _make_heartbeat_monitor(
    supervision_log_path: str,
    exit_reason_path: str,
    timeout_s: int,
    stop_event: _threading.Event,
    gen_num: int,
) -> _threading.Thread:
    """Watch supervision_log.txt and kill the subprocess if no new entry within timeout_s."""

    def _monitor():
        last_mtime = time.time()
        while not stop_event.is_set():
            stop_event.wait(timeout=5)
            if stop_event.is_set():
                break
            try:
                if os.path.exists(supervision_log_path):
                    mtime = os.path.getmtime(supervision_log_path)
                    if mtime > last_mtime:
                        last_mtime = mtime
            except Exception:
                pass
            if time.time() - last_mtime > timeout_s:
                reason = f"broken_gen: supervision heartbeat timeout ({timeout_s}s without supervision log update)"
                logger.warning(f"  [heartbeat] gen_{gen_num}: {reason}")
                try:
                    if not os.path.exists(exit_reason_path):
                        Path(exit_reason_path).write_text(reason)
                except Exception:
                    pass
                _kill_current_proc()
                break

    return _threading.Thread(target=_monitor, daemon=True, name=f"heartbeat-gen{gen_num}")

logger.info(f"Dataset directory: {ABS_DATASET_DIRECTORY}")
logger.info(f"Shared directory:  {ABS_SHARED_DIRECTORY}")

current_gen = -1
while True:
    current_gen += 1
    logger.info("=" * 80)
    logger.info(f"Generation {current_gen}")
    logger.info("=" * 80)

    # ── Validate package ──────────────────────────────────────────────────────

    current_gen_directory = os.path.abspath(f"{RUN_DIRECTORY}/gen_{current_gen}")
    target_agent_dir      = os.path.join(current_gen_directory, "target_agent")
    target_agent_path     = os.path.join(target_agent_dir, "target_agent.py")
    conf_yaml_path        = os.path.join(target_agent_dir, "conf.yaml")

    if not os.path.exists(target_agent_path):
        logger.error(f"  ✗ target_agent/target_agent.py not found: {target_agent_path}")
        sys.exit(1)
    if not os.path.exists(conf_yaml_path):
        logger.warning(f"  ⚠ target_agent/conf.yaml not found — agent will use its defaults")

    # ── Run target agent ──────────────────────────────────────────────────────

    logger.info(f"Running target agent: {target_agent_path}")
    stdout_log_file = os.path.join(current_gen_directory, "target_agent_stdout.log")
    generation_start_time = time.time()
    target_agent_success  = True
    target_agent_stdout   = ""
    target_agent_error_msg = ""

    try:
        python_exec = os.path.join(venv_dir, "bin", "python")
        state_file                 = os.path.join(current_gen_directory, "state.json")
        workspace_dir              = os.path.join(current_gen_directory, "workspace")
        solutions_dir_ta           = os.path.join(current_gen_directory, "solutions")
        exit_reason_path_ta        = os.path.join(current_gen_directory, "exit_reason.txt")
        agent_execution_path_ta    = os.path.join(current_gen_directory, "agent_execution.json")
        supervision_log_path_ta    = os.path.join(current_gen_directory, "supervision_log.txt")
        task_model_logs_dir_ta     = os.path.join(current_gen_directory, "task_model_logs")
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(solutions_dir_ta, exist_ok=True)
        os.makedirs(task_model_logs_dir_ta, exist_ok=True)

        _seed_arg = ""
        if current_gen == 0 and args.seed_solution:
            _seed_abs = os.path.abspath(args.seed_solution)
            if os.path.exists(_seed_abs):
                _seed_arg = f"--seed_solution {_seed_abs} "
            else:
                logger.warning(f"  ⚠ --seed_solution path not found: {_seed_abs} — skipping seed")

        command = (
            f"set -o pipefail; {python_exec} -u {target_agent_path} "
            f"--dataset_dir {ABS_DATASET_DIRECTORY} "
            f"--working_dir {workspace_dir} "
            f"--solutions_dir {solutions_dir_ta} "
            f"--exit_reason_path {exit_reason_path_ta} "
            f"--agent_execution_path {agent_execution_path_ta} "
            f"--supervision_log_path {supervision_log_path_ta} "
            f"--task_model_logs_dir {task_model_logs_dir_ta} "
            f"--shared_dir {ABS_SHARED_DIRECTORY} "
            f"--model {task_model} "
            f"--task_model_temperature {args.task_model_temperature} "
            f"--state_file {state_file} "
            f"--supervision_model {args.supervision_model} "
            f"--supervision_interval {args.supervision_interval} "
            f"--supervision_check_timeout {args.supervision_check_timeout_min * 60} "
            f"--gen0_evolve_duration {args.gen0_evolve_duration} "
            f"--exp_duration_seconds {int(args.exp_duration_min * 60)} "
            f"--current_gen {current_gen} "
            f"{_seed_arg}"
            f"2>&1 | tee {stdout_log_file}"
        )

        _hb_stop = _threading.Event()
        if current_gen > 0:
            _make_heartbeat_monitor(
                supervision_log_path=supervision_log_path_ta,
                exit_reason_path=exit_reason_path_ta,
                timeout_s=args.supervision_check_timeout_min * 60,
                stop_event=_hb_stop,
                gen_num=current_gen,
            ).start()

        try:
            return_code = _run_command(command)
        finally:
            _hb_stop.set()

        try:
            with open(stdout_log_file) as f:
                target_agent_stdout = f.read()
        except Exception:
            pass

        if return_code != 0:
            target_agent_success = False
            target_agent_error_msg = f"FAILED (exit code {return_code})"
            logger.error(f"  ✗ {target_agent_error_msg}")
        else:
            logger.info(f"  ✓ Generation {current_gen} completed successfully")

    except FileNotFoundError:
        logger.error(f"  ✗ Target agent not found: {target_agent_path}")
        sys.exit(1)
    except Exception as e:
        target_agent_success = False
        target_agent_error_msg = f"FAILED — {e}"
        logger.error(f"  ✗ {target_agent_error_msg}")
        try:
            with open(stdout_log_file) as f:
                target_agent_stdout = f.read()
        except Exception:
            pass

    generation_duration = time.time() - generation_start_time

    # ── Ensure exit_reason.txt exists ────────────────────────────────────────
    _exit_reason_path = os.path.join(current_gen_directory, "exit_reason.txt")
    if not os.path.exists(_exit_reason_path):
        _fallback_reason = (
            "broken_gen: no exit_reason.txt written (process killed or crashed without cleanup)"
            if not target_agent_success
            else "evolve: target agent completed without writing exit_reason.txt"
        )
        Path(_exit_reason_path).write_text(_fallback_reason)
        logger.warning(f"  ⚠ exit_reason.txt missing — wrote fallback: {_fallback_reason}")

    # ── Write state summary + validate ───────────────────────────────────────

    tree_summary = _write_state_summary(current_gen_directory)
    _, tree_is_valid = _build_rich_tree_summary(current_gen_directory, RUN_DIRECTORY)
    if not tree_is_valid:
        logger.warning(f"  ⚠ State validation failed — will trigger broken_gen")

    # ── stop / broken_gen detection ──────────────────────────────────────────

    should_stop_run    = False
    should_broken_gen  = not tree_is_valid
    exit_reason_path = os.path.join(current_gen_directory, "exit_reason.txt")
    exit_reason = ""
    if os.path.exists(exit_reason_path):
        exit_reason = Path(exit_reason_path).read_text().strip()
        # Format: "STATUS: reason" — extract the keyword before the colon
        exit_status = exit_reason.split(":")[0].strip()
        logger.info(f"  Exit reason: {exit_reason}")
        if exit_status == "stop":
            should_stop_run = True
        elif exit_status == "broken_gen":
            should_broken_gen = True
        # "evolve" = gen completed its budget normally; proceed to meta-agent

    if should_stop_run:
        logger.info("  → stop: terminating run.")
        break

    if should_broken_gen and current_gen > 0:
        # Capture broken gen info BEFORE _fallback_gen moves the directory
        _broken_agent_py = ""
        _broken_stdout_tail = target_agent_stdout[-3000:] if target_agent_stdout else ""
        _state_issue = "" if tree_is_valid else "state.json validation failed (broken structure or missing required node keys)"
        try:
            _bp = os.path.join(current_gen_directory, "target_agent", "target_agent.py")
            if os.path.exists(_bp):
                _broken_agent_py = Path(_bp).read_text(encoding="utf-8")
        except Exception:
            pass

        _fb_warning = _build_fallback_warning(
            gen_num=current_gen,
            error_msg=target_agent_error_msg,
            exit_reason=exit_reason,
            state_issue=_state_issue,
            broken_agent_py=_broken_agent_py,
            stdout_tail=_broken_stdout_tail,
        )

        logger.warning(f"  → broken_gen for gen_{current_gen}: moving to gen_broken, re-copying gen_{current_gen-1}")
        if _fallback_gen(current_gen, RUN_DIRECTORY):
            # Re-run meta-agent in the fresh copy
            _init_gen_state(current_gen_directory, task_id)
            best_score_fb, best_gen_fb = -1.0, current_gen - 1
            try:
                with open(os.path.join(os.path.join(RUN_DIRECTORY, f"gen_{current_gen-1}"), "state.json")) as _tf:
                    for _n in json.load(_tf).get("nodes", {}).values():
                        _r = _n.get("result")
                        if isinstance(_r, dict) and _r.get("score", -1) > best_score_fb:
                            best_score_fb = _r["score"]
                            best_gen_fb = _n.get("generation", current_gen - 1)
            except Exception:
                pass
            fb_prompt = _build_meta_agent_prompt(
                gen_num=current_gen,
                improvement_dir=current_gen_directory,
                best_score=best_score_fb,
                best_gen=best_gen_fb,
                tree_state_summary=_write_state_summary(current_gen_directory),
                best_agent_section="",
                fallback_warning=_fb_warning,
                prior_broken_history=_get_broken_gen_history(RUN_DIRECTORY, current_gen),
                exp_duration_min=args.exp_duration_min,
            )
            _run_meta_agent(fb_prompt, current_gen_directory, current_gen)
            tree_summary = _write_state_summary(current_gen_directory)
            logger.info(f"  ✓ Broken gen fixed — restarting gen_{current_gen} with repaired scaffold")
            continue
        else:
            logger.warning("  ⚠ Fallback failed — continuing without rollback")

    # ── Best-score lookup (shared by private eval + context) ─────────────────

    gen_best_sol   = None
    gen_best_score = -1.0
    best_score_so_far = -1.0
    best_gen_so_far   = current_gen
    try:
        with open(os.path.join(current_gen_directory, "state.json")) as _tf:
            for _n in json.load(_tf).get("nodes", {}).values():
                _r = _n.get("result")
                if not isinstance(_r, dict) or "score" not in _r:
                    continue
                _s = float(_r["score"])
                if _s > best_score_so_far:
                    best_score_so_far = _s
                    best_gen_so_far = _n.get("generation", current_gen)
                if _n.get("generation") == current_gen and _s > gen_best_score and _n.get("solution_path"):
                    gen_best_score = _s
                    gen_best_sol = _n["solution_path"]
    except Exception:
        pass

    prev_gen_best_score: float | None = None
    if current_gen > 0:
        try:
            with open(os.path.join(RUN_DIRECTORY, f"gen_{current_gen - 1}", "state.json")) as _tf:
                for _n in json.load(_tf).get("nodes", {}).values():
                    _r = _n.get("result")
                    if isinstance(_r, dict) and "score" in _r:
                        _s = float(_r["score"])
                        if prev_gen_best_score is None or _s > prev_gen_best_score:
                            prev_gen_best_score = _s
        except Exception:
            pass

    # ── Private evaluation ────────────────────────────────────────────────────

    private_eval = os.path.join(task_dir, "data/private/evaluate.py")
    if os.path.exists(private_eval):
        if not gen_best_sol or not os.path.exists(gen_best_sol):
            logger.warning(f"  [private] No valid solution in state.json for gen_{current_gen} — skipping")
        else:
            priv_work_dir = os.path.abspath(
                os.path.join(RUN_DIRECTORY, "private_scores", f"gen_{current_gen}")
            )
            os.makedirs(priv_work_dir, exist_ok=True)
            logger.info(f"  [private] evaluating gen_{current_gen}: {gen_best_sol}")
            rc = _run_command(f"{python_exec} {private_eval} {gen_best_sol}", cwd=priv_work_dir)
            priv_result_path = os.path.join(priv_work_dir, "private_result.json")
            if os.path.exists(priv_result_path):
                with open(priv_result_path) as f:
                    priv_result = json.load(f)
                logger.info(f"  [private] gen_{current_gen}: score={priv_result.get('score', '?')}")
            else:
                logger.warning(f"  [private] gen_{current_gen}: private_result.json not written (exit code {rc})")

    # ── Regenerate plot ───────────────────────────────────────────────────────

    _plot_path = _plot_scores(RUN_DIRECTORY, task_name=os.path.basename(task_dir))
    if _plot_path:
        logger.info(f"  [plot] Updated: {_plot_path}")

    # ── Prepare next generation: copy forward + meta-agent ───────────────────

    next_gen = current_gen + 1
    next_gen_directory = os.path.abspath(f"{RUN_DIRECTORY}/gen_{next_gen}")

    _copy_gen_forward(current_gen_directory, next_gen_directory)

    _append_context(
        next_gen_directory, current_gen, tree_summary,
        success=target_agent_success,
        duration=generation_duration,
        best_score=gen_best_score if gen_best_score > 0 else None,
        prev_best_score=prev_gen_best_score,
        exit_reason=exit_reason,
        improvement_md_path=os.path.join(current_gen_directory, "improvement.md"),
        meta_model=meta_model,
    )

    best_agent_section = (
        f"Best score so far: {best_score_so_far:.4f} (gen {best_gen_so_far}).\n"
        f"Read state_summary.md for top nodes and solution paths."
    )

    logger.info(f"Running meta-agent for generation {next_gen}")
    prompt = _build_meta_agent_prompt(
        gen_num=next_gen,
        improvement_dir=next_gen_directory,
        best_score=best_score_so_far,
        best_gen=best_gen_so_far,
        tree_state_summary=tree_summary,
        best_agent_section=best_agent_section,
        prior_broken_history=_get_broken_gen_history(RUN_DIRECTORY, next_gen),
        exp_duration_min=args.exp_duration_min,
    )

    if not _run_meta_agent(prompt, next_gen_directory, next_gen):
        logger.error(f"  ✗ Meta-agent failed for gen_{next_gen} — aborting")
        sys.exit(1)

    logger.info(f"  ✓ Meta-agent done for gen_{next_gen}")


# ========================
# SECTION 6: Finalize
# ========================

plot_path = _plot_scores(RUN_DIRECTORY, task_name=os.path.basename(task_dir))
if plot_path:
    logger.info(f"Final plot: {plot_path}")

final_state_path = os.path.join(os.path.abspath(f"{RUN_DIRECTORY}/gen_{current_gen}"), "state.json")
if os.path.exists(final_state_path):
    try:
        tree     = json.load(open(final_state_path))
        best_id  = tree.get("best_node_id")
        best_node = tree["nodes"].get(best_id) if best_id else None
        if best_node and isinstance(best_node.get("result"), dict):
            logger.info(f"Best result: node={best_id}  score={best_node['result'].get('score')}")
            if best_node.get("solution_path"):
                logger.info(f"Best solution: {best_node['solution_path']}")
    except Exception:
        pass

logger.info("=" * 80)
logger.info(f"Orchestrator completed. Last generation: {current_gen}.")
logger.info(f"Results: {RUN_DIRECTORY}")
logger.info("=" * 80)
