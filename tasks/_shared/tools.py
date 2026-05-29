"""
tools.py — sandboxed tool implementations for target agents.

Usage in your target_agent.py:
    import sys
    sys.path.insert(0, shared_dir)   # shared_dir passed via --shared_dir
    from tools import bash, read_file, write_file, submit_solution

Path restrictions:
  - bash:            working_dir, solutions_dir (read), dataset_dir, system paths
  - read_file:       working_dir, solutions_dir, or dataset_dir
  - write_file:      working_dir only (use submit_solution to submit a solution)
  - submit_solution: writes solutions/{uid}.py, evaluates, registers in state.json
"""

from __future__ import annotations

import json as _json
import os
import re
import signal
import subprocess
import sys as _sys
import uuid as _uuid
from pathlib import Path

SYSTEM_PATHS = ("/usr", "/bin", "/lib", "/etc", "/tmp", "/dev", "/proc", "/sys")


def _bash_paths_allowed(command: str, working_dir: str, dataset_dir: str, solutions_dir: str | None = None) -> bool:
    allowed = [Path(working_dir).resolve(), Path(dataset_dir).resolve()]
    if solutions_dir:
        allowed.append(Path(solutions_dir).resolve())
    for m in re.finditer(r"(?<!\w)(\/[^\s\"'|;&><,()]+)", command):
        p = Path(m.group(1)).resolve()
        if any(p.is_relative_to(d) for d in allowed):
            continue
        if any(str(p).startswith(s) for s in SYSTEM_PATHS):
            continue
        return False
    return True


def bash(command: str, *, working_dir: str, dataset_dir: str, solutions_dir: str | None = None) -> str:
    if not _bash_paths_allowed(command, working_dir, dataset_dir, solutions_dir):
        return "[ERROR] bash command references a path outside working_dir, solutions_dir, or dataset_dir"
    try:
        proc = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, cwd=working_dir,
        )
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            proc.wait()
            return "[ERROR] timed out"
        out = stdout + (f"\n[stderr]\n{stderr}" if stderr else "")
        return out.strip() or "(no output)"
    except Exception as e:
        return f"[ERROR] {e}"


def _resolve_path(path: str, base_dir: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(base_dir) / p
    return p.resolve()


def read_file(path: str, *, working_dir: str, dataset_dir: str, solutions_dir: str | None = None) -> str:
    try:
        p = _resolve_path(path, working_dir)
        allowed = [Path(working_dir).resolve(), Path(dataset_dir).resolve()]
        if solutions_dir:
            allowed.append(Path(solutions_dir).resolve())
        if not any(p.is_relative_to(d) for d in allowed):
            return "[ERROR] read_file path must be inside working_dir, solutions_dir, or dataset_dir"
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[ERROR] {e}"


def write_file(path: str, content: str, *, working_dir: str) -> str:
    try:
        p = _resolve_path(path, working_dir)
        if not p.is_relative_to(Path(working_dir).resolve()):
            return "[ERROR] write_file path must be inside working_dir — to submit a solution use submit_solution instead"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"
    except Exception as e:
        return f"[ERROR] {e}"


def submit_solution(
    code: str,
    *,
    solutions_dir: str,
    working_dir: str,
    evaluate_path: str,
    state_file: str | None = None,
    current_gen: int = 0,
    write_node_fn=None,
) -> str:
    """Write code to solutions/{uid}.py, evaluate it, register in state.json. Returns JSON."""
    uid = str(_uuid.uuid4())[:8]
    sol_path = os.path.join(solutions_dir, f"{uid}.py")

    try:
        Path(sol_path).write_text(code, encoding="utf-8")
    except Exception as e:
        return _json.dumps({"error": f"Failed to write solution: {e}", "score": None, "uid": uid})

    try:
        proc = subprocess.run(
            [_sys.executable, evaluate_path, sol_path],
            capture_output=True, text=True, cwd=working_dir, timeout=300,
        )
        full_out = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    except Exception as e:
        return _json.dumps({"error": f"Evaluation subprocess failed: {e}", "score": None, "uid": uid})

    if "RESULT_JSON:" not in full_out:
        tail = full_out[-500:] if full_out else "(no output)"
        return _json.dumps({"error": "No RESULT_JSON in evaluator output", "score": None, "uid": uid, "output_tail": tail})

    try:
        json_str = full_out.split("RESULT_JSON:", 1)[1].split("\n", 1)[0].strip()
        result = _json.loads(json_str)
    except Exception as e:
        return _json.dumps({"error": f"RESULT_JSON parse failed: {e}", "score": None, "uid": uid})

    node_id = None
    if state_file and write_node_fn:
        try:
            node_id = write_node_fn(state_file, sol_path, result, current_gen)
        except Exception:
            pass

    status = f"score={result['score']:.4f}" if result.get("score") is not None and not result.get("error") else f"error={str(result.get('error', ''))[:80]}"
    return _json.dumps({
        "uid": uid,
        "score": result.get("score"),
        "error": result.get("error"),
        "node_id": node_id,
        "lower_is_better": result.get("lower_is_better"),
        "_status": status,
    })
