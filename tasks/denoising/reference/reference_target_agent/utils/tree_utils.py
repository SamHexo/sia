"""Tree state utilities shared by all SIA target-agent generations.

These helpers provide a stable interface for reading and writing state.json.
The meta-agent can import them, extend them, or replace them — as long as
nodes written to state.json preserve the required keys:
  - uuid       (str)
  - generation (int)
  - result     (dict with at least "score": float)
"""

import json
import logging
import os
import shutil
import uuid as _uuid_mod
from pathlib import Path

logger = logging.getLogger(__name__)


def write_tree_node(
    state_file: str,
    solution_path: str,
    result: dict,
    generation: int,
) -> str | None:
    """Append an evaluated solution as a node in state.json.

    If solution_path is already inside solutions/, uses it in place (LLM wrote
    there directly) and takes the filename stem as the UUID.
    Otherwise copies the file to gen_X/solutions/{uuid}.py.
    Updates state.json atomically. Returns the node_id on success, None if
    state_file is missing.
    """
    if not os.path.exists(state_file):
        return None

    solutions_dir = os.path.join(os.path.dirname(state_file), "solutions")
    os.makedirs(solutions_dir, exist_ok=True)

    abs_sol = os.path.abspath(solution_path)
    abs_solutions = os.path.abspath(solutions_dir)

    if abs_sol.startswith(abs_solutions + os.sep):
        # Already in solutions/ — LLM wrote here directly; filename stem is the UUID
        node_uuid = Path(abs_sol).stem
        uuid_sol_path = abs_sol
    else:
        # File is elsewhere — generate UUID and copy
        node_uuid = str(_uuid_mod.uuid4())
        uuid_sol_path = os.path.join(solutions_dir, f"{node_uuid}.py")
        try:
            shutil.copy2(solution_path, uuid_sol_path)
        except Exception as e:
            logger.warning(f"Could not copy solution: {e}")
            uuid_sol_path = solution_path

    with open(state_file) as f:
        tree = json.load(f)

    # iteration_id = sequential index within this generation; always override
    # (evaluate.py defaults --iteration_id to 0, so we must not use setdefault)
    iteration_id = sum(
        1 for n in tree["nodes"].values()
        if n.get("generation") == generation and isinstance(n.get("result"), dict)
    )
    result = dict(result)
    result["iteration_id"] = iteration_id

    is_buggy = bool(result.get("error"))
    node_id = f"node_{len(tree['nodes']):04d}"
    node = {
        "id": node_id,
        "parent": tree["root_id"],
        "children": [],
        "uuid": node_uuid,
        "generation": generation,
        "solution_path": uuid_sol_path,
        "result": result,
        "visits": 1,
        "status": "buggy" if is_buggy else "evaluated",
        "metadata": {},
    }
    tree["nodes"][node_id] = node
    tree["nodes"][tree["root_id"]]["children"].append(node_id)

    if not is_buggy:
        cur_best = tree.get("best_node_id")
        cur_best_score = -1.0
        if cur_best and cur_best in tree["nodes"]:
            cur_best_score = (tree["nodes"][cur_best].get("result") or {}).get("score", -1.0)
        if result.get("score", 0.0) > cur_best_score:
            tree["best_node_id"] = node_id

    tmp = state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tree, f, indent=2)
    os.replace(tmp, state_file)
    logger.info(f"Node written: {node_id} uuid={node_uuid[:8]} score={result.get('score', 0.0):.4f}")
    return node_id


def read_state(state_file: str) -> dict:
    """Load state.json. Raises on missing file or parse error."""
    with open(state_file) as f:
        return json.load(f)


def get_best_node(state_file: str) -> tuple[dict | None, float]:
    """Return (best_node_dict, best_score). Returns (None, -1.0) if no evaluated nodes."""
    try:
        tree = read_state(state_file)
    except Exception:
        return None, -1.0
    best_id = tree.get("best_node_id")
    if best_id and best_id in tree.get("nodes", {}):
        node = tree["nodes"][best_id]
        score = float((node.get("result") or {}).get("score", -1.0))
        return node, score
    best_node, best_score = None, -1.0
    for n in tree.get("nodes", {}).values():
        r = n.get("result")
        if isinstance(r, dict) and "score" in r:
            s = float(r["score"])
            if s > best_score:
                best_score = s
                best_node = n
    return best_node, best_score


def get_generation_nodes(state_file: str, generation: int) -> list[dict]:
    """Return all evaluated nodes produced by a specific generation."""
    try:
        tree = read_state(state_file)
    except Exception:
        return []
    return [
        n for n in tree.get("nodes", {}).values()
        if n.get("generation") == generation and isinstance(n.get("result"), dict)
    ]


def summarize_tree(state_file: str) -> str:
    """Return a compact human-readable summary of the tree state.

    Raises ValueError on structural problems (broken JSON, missing required keys).
    """
    if not state_file:
        raise ValueError("No state_file provided")
    if not os.path.exists(state_file):
        raise ValueError(f"state.json not found: {state_file}")
    try:
        with open(state_file) as f:
            tree = json.load(f)
    except Exception as e:
        raise ValueError(f"state.json parse error: {e}") from e
    if "nodes" not in tree:
        raise ValueError("state.json missing 'nodes' key")

    clean, buggy = [], []
    for nid, n in tree["nodes"].items():
        if not isinstance(n.get("result"), dict):
            continue
        result = n["result"]
        if "score" not in result:
            raise ValueError(f"Node {nid} has result but missing 'score' key")
        if "generation" not in n:
            raise ValueError(f"Node {nid} missing 'generation' key")
        if n.get("status") == "buggy":
            buggy.append(n)
        else:
            clean.append((result["score"], n))

    if not clean and not buggy:
        return "No evaluated nodes yet."
    clean.sort(key=lambda x: x[0], reverse=True)
    best = clean[0][0] if clean else 0.0
    gen_counts: dict[int, int] = {}
    for _, n in clean:
        g = n["generation"]
        gen_counts[g] = gen_counts.get(g, 0) + 1
    gen_summary = "  ".join(f"gen{g}:{c}" for g, c in sorted(gen_counts.items()))
    buggy_note = f"  {len(buggy)} buggy." if buggy else ""
    return f"{len(clean)} nodes evaluated. Best score: {best:.4f}. [{gen_summary}]{buggy_note}"
