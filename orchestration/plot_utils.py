import os
import re
import json
import glob


def plot_scores(run_directory: str, task_name: str = "") -> str | None:
    """
    Plot all public scores (every evaluated iteration, per gen) and the private
    score of the best-per-gen solution aligned to its iteration's x position.

    X-axis: global iteration index (continuous across gens), with vertical dashed
    lines at gen boundaries and gen labels as x-tick labels.

    - Blue dots + line:    all public scores for every evaluated solution
    - Red diamonds:        private score of the best solution per gen, placed at
                           the x position of that solution's iteration

    Saves private_score.png to run_directory/private_scores/. Returns the path or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return None

    # ── Find latest state.json (cumulative: contains all gens) ────────────────
    latest_state = None
    g = 0
    while True:
        p = os.path.join(run_directory, f"gen_{g}", "state.json")
        if not os.path.exists(p):
            break
        latest_state = p
        g += 1

    if latest_state is None:
        return None

    try:
        with open(latest_state) as f:
            tree = json.load(f)
    except Exception:
        return None

    # ── Collect evaluated nodes per gen ───────────────────────────────────────
    # gen → [(iteration_id, node_id, score), ...]
    gen_nodes: dict[int, list] = {}
    lower_is_better = False

    for node_id, node in tree.get("nodes", {}).items():
        result = node.get("result")
        if not isinstance(result, dict) or "score" not in result:
            continue
        gen = node.get("generation", 0)
        score = float(result["score"])
        lower_is_better = result.get("lower_is_better", lower_is_better)
        iter_id = result.get("iteration_id")  # may be None for older nodes
        gen_nodes.setdefault(gen, []).append((iter_id, node_id, score))

    if not gen_nodes:
        return None

    # Sort within each gen: by iteration_id if present, else by node_id suffix
    sorted_gens = sorted(gen_nodes.keys())
    for gen in sorted_gens:
        nodes = gen_nodes[gen]
        if all(it is not None for it, _, _ in nodes):
            nodes.sort(key=lambda x: x[0])
        else:
            def _key(x):
                m = re.search(r"(\d+)$", x[1])
                return int(m.group(1)) if m else 0
            nodes.sort(key=_key)
            # Back-fill iteration_id from sort order
            gen_nodes[gen] = [(i, nid, sc) for i, (_, nid, sc) in enumerate(nodes)]

    # ── Compute global x offsets per gen ──────────────────────────────────────
    gen_offsets: dict[int, int] = {}
    offset = 0
    for gen in sorted_gens:
        gen_offsets[gen] = offset
        offset += len(gen_nodes[gen])
    total_iterations = offset

    if total_iterations == 0:
        return None

    # ── Collect private scores ─────────────────────────────────────────────────
    # private_scores/gen_N/private_result.json
    private_points: list[tuple[float, float]] = []  # (global_x, private_score)

    for priv_path in sorted(glob.glob(
        os.path.join(run_directory, "private_scores", "gen_*", "private_result.json")
    )):
        # path: .../private_scores/gen_N/private_result.json
        gen_dir_name = os.path.basename(os.path.dirname(priv_path))
        m = re.fullmatch(r"gen_(\d+)", gen_dir_name)
        if not m:
            continue
        gen = int(m.group(1))
        if gen not in gen_nodes:
            continue
        try:
            with open(priv_path) as f:
                d = json.load(f)
        except Exception:
            continue
        priv_score = d.get("score")
        if priv_score is None or d.get("error"):
            continue

        # Find the best public node for this gen → its iteration_id → global x
        best_iter, best_score = None, (-1e9 if not lower_is_better else 1e9)
        for it_id, _, sc in gen_nodes[gen]:
            if (not lower_is_better and sc > best_score) or (lower_is_better and sc < best_score):
                best_score = sc
                best_iter = it_id
        if best_iter is None:
            continue
        private_points.append((gen_offsets[gen] + best_iter, float(priv_score)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    direction = "lower" if lower_is_better else "higher"
    fig, ax = plt.subplots(figsize=(max(12, total_iterations * 0.35 + 2), 5))

    # Public scores: all iterations
    all_xs = []
    all_ys = []
    for gen in sorted_gens:
        for it_id, _, sc in gen_nodes[gen]:
            all_xs.append(gen_offsets[gen] + it_id)
            all_ys.append(sc)

    # Sort by x so the connecting line is monotone
    sorted_pub = sorted(zip(all_xs, all_ys))
    all_xs = [p[0] for p in sorted_pub]
    all_ys = [p[1] for p in sorted_pub]

    ax.plot(all_xs, all_ys, linewidth=1.0, color="#4C72B0", alpha=0.4, zorder=2)
    ax.scatter(all_xs, all_ys, s=30, color="#4C72B0", alpha=0.85, zorder=3, label="Public")

    # Private score diamonds at best-iteration position
    if private_points:
        px = [p[0] for p in private_points]
        py = [p[1] for p in private_points]
        ax.scatter(px, py, s=90, marker="D", color="#DD4444", zorder=5,
                   edgecolors="darkred", linewidths=0.8, label="Private (best of gen)")

    # Gen boundaries and x-tick labels
    xtick_positions = []
    xtick_labels = []
    for gen in sorted_gens:
        n = len(gen_nodes[gen])
        center_x = gen_offsets[gen] + (n - 1) / 2.0
        xtick_positions.append(center_x)
        xtick_labels.append(f"gen {gen}")
        if gen_offsets[gen] > 0:
            ax.axvline(gen_offsets[gen] - 0.5, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.4, zorder=1)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel("Iteration (grouped by generation)")
    ax.set_ylabel(f"Score ({direction} is better)")
    ax.set_title(f"Score evolution — {task_name or os.path.basename(run_directory)}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    plot_dir = os.path.join(run_directory, "private_scores")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "private_score.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


# Alias kept for any external callers
plot_private_scores = plot_scores
