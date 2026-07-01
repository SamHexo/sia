# SIA with Persistent Search State: When the Artefact Is a Solution, Not an Agent

**Samuel Verboomen**  
Hexo Labs

---

**Keywords:** Self-Improving Agents, Tree Search, MCTS, Scaffold Evolution, Compute Allocation, Algorithmic Benchmarks

---

## Abstract

Self-improving agent frameworks iterate a meta-agent loop: run a task-specific scaffold, observe results, rewrite the scaffold, repeat. This loop is well-suited when the final artefact is the scaffold itself — an evolved agent that handles legal queries or customer dialogue. But for a large and practically important class of tasks — algorithm optimisation, ML engineering, scientific computing — the final deliverable is a *solution file*: a piece of code that solves an instance, not an agent that navigates one. In these settings, the scaffold is a search engine, and the tree of explored solutions is the primary asset. Discarding that tree at each generation boundary wastes the computation of every prior generation. We describe a variant of SIA that carries the full search tree forward across generations, so that an improved scaffold inherits, rather than abandons, the exploration history of its predecessors. We characterise the benchmark landscape along the agent-vs-solution axis, present an architecture — persistent PUCT state, generation-scoped solution storage, and a supervision loop — that implements this idea, and report preliminary results on the scRNA-seq denoising task: 254 candidate solutions accumulated across six generations in a single 12-hour run, reaching a best public score of 0.653 from a baseline of 0.614.

---

## 1. Introduction

### 1.1 The standard SIA loop and its implicit assumption.

The SIA loop (Hebbar et al., 2026) iterates three components: a Meta-Agent proposes an initial scaffold, a Target Agent executes it against a verifier, and a Feedback-Agent synthesises an improved scaffold from the observed trajectory. This cycle repeats until a compute budget is exhausted. The implicit assumption is that the scaffold is the final product: at the end of generation $g$, the improved scaffold $A_{g+1}$ is the deliverable, and the rollouts that produced it are merely training signal.

This assumption holds cleanly on tasks like LawBench, where the benchmark measures the classification accuracy of the agent itself, or tau2-bench, where it measures multi-turn customer service quality. In both cases, improving the agent's decision-making strategy *is* the goal.

### 1.2 A different class of tasks.

Many benchmarks are structured differently. On GPU kernel optimisation, the verifier times a submitted Triton kernel and returns a runtime. On scRNA-seq denoising, it scores a `custom_denoise` function against held-out expression data. On MLE-bench, it evaluates a submitted training pipeline on a held-out test split. In all these cases, the agent is a search engine and the solution it produces — a file, a kernel, a function — is the deliverable. The agent itself is discarded when the run ends; only the best solution survives.

This distinction has a concrete consequence for how compute should be allocated.

### 1.3 The restart problem.

Suppose a 10-hour compute budget. A naive self-improving loop spends 1 hour on generation 0 (a weak scaffold), then 8 hours iterating the scaffold, then 1 hour running the final, polished scaffold. The 8 hours of scaffold iteration did not produce solutions — they produced a better search strategy. But when that better strategy runs for only 1 hour, it starts from scratch. It re-explores directions the weak scaffold already visited and discards the 1 hour of solutions it produced.

The claim of this paper is: for solution-artefact tasks, **a mediocre scaffold running for 10 hours dominates a polished scaffold running for 1 hour after 9 hours of meta-iteration** — unless the improved scaffold inherits the search state of its predecessors. Persistent tree search resolves this by carrying the explored node tree, all evaluated solutions, and their scores across the generation boundary. The improved scaffold continues from where the previous one stopped.

---

## 2. A Taxonomy of Self-Improvement Benchmarks

The distinction between agent artefacts and solution artefacts is the primary axis along which self-improvement benchmarks differ. Table 1 maps a representative set of benchmarks across four dimensions: the domain, the unit of evaluation, the final artefact, and the appropriate search paradigm.

**Table 1. Benchmark taxonomy along the artefact axis.**

| Benchmark | Domain | Evaluation unit | Final artefact | Search paradigm |
|---|---|---|---|---|
| LawBench | Chinese legal classification | Per-instance top-1 accuracy | Evolved agent | Instance rollout |
| tau2-bench | Customer service tool use | Multi-turn episode reward | Evolved agent | Dialogue episode |
| SWE-bench | Software bug fixing | Patch apply + test pass | Evolved agent | Code editing agent |
| GPQA | Graduate-level science Q&A | Chain-of-thought accuracy | Evolved agent | Reasoning rollout |
| Denoising (scRNA-seq) | Computational biology | MSE + Poisson on held-out tissue | **Solution file** | MCTS solution tree |
| TriMul CUDA | GPU kernel optimisation | Kernel runtime on H100 | **Solution file** | MCTS solution tree |
| MLE-bench | ML engineering | Held-out test metric | **Solution file** | MCTS solution tree |
| Spaceship Titanic | Tabular ML (Kaggle) | Classification accuracy | **Solution file** | MCTS solution tree |

The top four rows share a common structure: the benchmark measures the behaviour of a deployed agent. The scaffold that wraps the LLM — its prompt, tool-dispatch logic, retry policy — is what the meta-agent refines, and the scaffold at the final generation is the deliverable. Carrying forward rollout histories across generations is useful for feedback but not strictly necessary: the agent's performance depends on its current code, not on which trajectories an earlier version of that code produced.

The bottom four rows share a different structure. The benchmark measures a submitted artefact — a Python function, a Triton kernel, a fitted model. The agent that produced it is irrelevant once the artefact is submitted. What matters is that the artefact is the best one found in the available compute budget. In these settings, every evaluated solution is a data point in a search tree, and discarding that tree at a generation boundary is equivalent to throwing away explored branches of a search.

---

## 3. Architecture

### 3.1 Generation structure.

Each generation $g$ occupies a directory `gen_g/` containing:
- `target_agent/` — the scaffold package, modified by the meta-agent from the previous generation
- `state.json` — the shared search tree, accumulating all evaluated nodes across all prior generations
- `solutions/` — the evaluated solution files, pruned to the subset referenced by `state.json`
- `state_summary.md` — a human-readable rendering of the tree, passed to the meta-agent and supervisor
- `context.md` — a cumulative log of all generation decisions and improvement reports

Only these artefacts are carried forward. Generation-specific files (execution logs, workspace scratch, exit reason) are not copied.

Generation 0 is a reference scaffold: a classical MCTS loop over the solution space, with no meta-agent involvement, that runs until a fixed time budget is reached. It serves as the initial tree population that all subsequent generations build on.

### 3.2 The state schema.

`state.json` encodes the search tree as a flat node dictionary with explicit parent/child pointers. Each evaluated node carries a small set of required fields alongside a free-form `metadata` dict:

```json
{
  "version": "0.2",
  "root_id": "node_0000",
  "best_node_id": "node_0211",
  "nodes": {
    "node_0001": {
      "id":             "node_0001",
      "parent":         "node_0000",
      "children":       [],
      "uuid":           "3de2f47b",
      "generation":     1,
      "solution_path":  ".../gen_1/solutions/3de2f47b.py",
      "result":         {"score": 0.6150, "iteration_id": 0, ...},
      "visits":         2,
      "status":         "evaluated",
      "metadata":       {}
    }
  }
}
```

The **required fields** for each evaluated node are:

| Field | Type | Role |
|---|---|---|
| `uuid` | string | Filename stem of the solution in `solutions/` — the stable identifier that links a node to its artefact on disk |
| `generation` | int | Which generation produced this node; used by the meta-agent and supervisor to attribute score trends to specific scaffold versions |
| `result.score` | float | The scalar score returned by the verifier; the only field the PUCT selection reads |
| `parent` / `children` | string / list | Tree topology; children are appended atomically by `write_tree_node` |
| `visits` | int | Number of times this node was selected as a parent for expansion; persists across generations |
| `status` | string | One of `"root"`, `"evaluated"`, or `"buggy"` |

The `metadata` dict is entirely free-form. The meta-agent can use it to store anything that helps its search strategy — solution family tags, estimated algorithmic complexity, partial results along secondary metrics, flags marking nodes as seed candidates for a new approach. It is carried forward unchanged across generations.

The `visits` counter persists across generations: a node selected twice in generation 3 still carries that count into generation 4, so its accumulated exploration bonus is preserved. When copying forward, the orchestrator prunes `solutions/` to the set of UUIDs referenced in `state.json`, preventing unbounded storage growth.

### 3.3 Generation 0 and the reference search policy.

Generation 0 ships with a standard PUCT selection policy as a reference implementation. PUCT balances exploitation (selecting high-scoring nodes) with exploration (revisiting under-sampled regions) and gives a reasonable starting point without any domain knowledge. Its single tunable parameter, the exploration constant $c_\text{puct}$, is exposed in `conf.yaml`.

This policy is deliberately minimal. Its purpose is to populate the tree with enough evaluated nodes that the meta-agent's first feedback call has meaningful signal to work with. The meta-agent is free to replace it entirely in generation 1 — the only obligation is to keep writing nodes into `state.json` via `write_tree_node`.

### 3.4 The scaffold package and meta-agent freedom.

Unlike the original SIA harness, where the meta-agent edits a single flat file, this variant treats `target_agent/` as a full Python package. The only two files the orchestrator requires at startup are `target_agent/target_agent.py` and `conf.yaml`; everything else is the meta-agent's to design freely.

In practice this means the meta-agent can:

- **Add sub-agents.** A specialised inner agent can be responsible for a subtask — proposing mutations, running a cheap proxy evaluator, summarising the solution tree. The outer loop in `target_agent.py` orchestrates them.
- **Add utility modules.** Domain-specific helpers (`domain_analyzer.py`, `feature_builder.py`), prompt template files, or pre-computed data caches can be added as sibling modules inside `target_agent/`.
- **Restructure the search loop entirely.** PUCT can be replaced by beam search, random restarts, evolution strategies, or any other policy. The only contract with the orchestrator is the `state.json` write interface and the `while True` loop in `main()`.
- **Tag solutions with families.** By writing structured entries into the `metadata` field of each node, the meta-agent can group solutions by algorithmic approach (e.g., `{"family": "graph-diffusion", "variant": "knn-weighted"}`), then steer future expansions toward under-explored families.
- **Extend `conf.yaml` freely.** Any YAML structure is valid. The meta-agent can add new sections for hyperparameter sweeps, prompt variants, or inner-agent configuration — anything it wants to make tunable between generations without modifying source code.
- **Install packages.** New Python libraries can be installed into the run's virtual environment at any point and are immediately available to all subsequent generations.

The hard constraints are few: `target_agent.py` must exist and contain a `while True` loop in `main()`; solutions must be registered via `write_tree_node`; and the agent must load the existing `state.json` at startup and continue from it rather than reinitialising. Beyond these three invariants, the meta-agent has full discretion over the package structure and search logic.

### 3.5 The supervision loop.

A supervision agent runs inside the target agent every $k$ evaluated nodes (default: $k = 5$). It receives the current `state_summary.md`, the elapsed time, and the remaining global budget, and emits one of three decisions:

- **CONTINUE** — the scaffold is still making progress; keep running
- **EVOLVE** — stop this generation and invoke the meta-agent; the scaffold has plateaued or a structural change is warranted
- **STOP** — terminate the run; the global budget is exhausted or a fatal error makes further progress impossible

The supervisor is the only component that has visibility into both the search tree state and the global time budget. Its primary signals are the per-generation score improvement curve and the ratio of new solutions to revisited nodes. When the selection policy begins cycling back to nodes already visited many times, and per-generation marginal improvement is low, the supervisor should prefer EVOLVE over CONTINUE even if absolute scores are still rising.

Generation 0 uses a simpler rule: it auto-evolves after a fixed time (`gen0_evolve_duration`, default 15 minutes). This ensures the initial tree has enough nodes to make the meta-agent's first feedback call meaningful.

**Broken generations.** A generation is classified as `broken_gen` in two situations: a heartbeat timeout (no entry written to the supervision log for 1200 seconds, indicating the agent has silently hung), or a structural failure where the generation does not meet the invariants required for the next generation to start correctly — for instance, a malformed `state.json` that would corrupt the tree, or a scaffold that exits immediately without writing any nodes. In both cases the orchestrator moves the directory to `gen_broken/`, re-copies the last working generation, and re-runs the meta-agent with an explicit warning about the failure. The lost computation is limited to one generation; all prior tree state is preserved.

---

## 4. Preliminary Results: Denoising (run 22)

We report results on the scRNA-seq denoising task from the SIA benchmark suite. The task asks for a `custom_denoise(X)` function that imputes dropout noise in single-cell RNA sequencing data. The development dataset is pancreatic islet cells; the final score is computed on a private held-out dataset (PBMC blood cells and Tabula multi-tissue atlas) that neither the target agent nor the meta-agent can access during the run. The private evaluation runs automatically at the end of each generation, but its result is not fed back into the loop — it exists purely as an external validity check, not as a training signal. The metric is a normalised average of MSE and Poisson deviation, higher is better. MAGIC, the reference tool, scores approximately 0.64 on the private sets.

**Run 22 is presented as an illustrative example of tree growth across generations, not as a performance benchmark.** The private scores show no meaningful improvement over the development scores, likely due to insufficient diversity in the random seeds used during search — the solutions found by the target agent clustered around similar approaches, and the private dataset exposed this lack of generalisation. A properly calibrated run would require more explicit diversity pressure in the search policy. The data below should be read as evidence that the tree accumulation mechanism works as intended, not as a claim about the quality of the solutions found.

**Table 2. Run 22 — cumulative tree growth and best score per generation (pancreas dev set).**

| Generation | New solutions | Cumulative nodes | Best score (all time) | Best score (this gen) |
|---|---|---|---|---|
| 0 (reference MCTS) | 5 | 5 | 0.614 | 0.614 |
| 1 | 20 | 25 | 0.617 | 0.617 |
| 2 | 27 | 52 | 0.617 | 0.613 |
| 3 | 27 | 79 | 0.617 | 0.617 |
| 4 | 50 | 129 | 0.634 | 0.634 |
| 5 | 60 | 189 | 0.635 | 0.635 |
| 6 | 65 | **254** | **0.653** | **0.653** |

![Run 22 — public vs private score evolution across generations](../runs/run_22/private_scores/private_score.png)

**Figure 1. Public vs private score divergence in run 22.** The blue line tracks the best public score (pancreas dev set) per generation; the red dashed line tracks the private score (PBMC + Tabula, computed by the orchestrator at end of each generation, never fed back to the agents). The public score improves monotonically from 0.614 to 0.653. The private score stays flat around 0.640 throughout, then drops to 0.627 at the final generation (Δ = −0.014), indicating that the late-generation scaffolds over-specialised to the development dataset. The bottom panel shows, per generation, the number of nodes explored, the number of solution families tracked in `metadata`, and the scaffold name as labelled by the meta-agent.

The total run lasted 720 minutes across 7 generations. Three structural observations are worth noting. First, generations 1–3 show only marginal improvement over the reference scaffold, but they contribute 74 evaluated nodes that remain in the tree. When the meta-agent produces a substantially improved scaffold at generation 4 — one that restructures the search strategy — that scaffold immediately benefits from the full prior exploration. Generation 4's best score (0.634) is reached not by re-discovering solutions the earlier scaffolds found, but by the selection policy steering the new scaffold toward unexplored regions adjacent to the best prior nodes.

Second, the generation-6 leap (0.635 → 0.653) is the largest single-generation improvement in the run. At that point the tree has 189 nodes, and the improved scaffold's selection policy has a rich map of the solution space to navigate. A restart from scratch would have given generation 6 only the 65 solutions it produced in that generation, with no prior context.

Third, the run terminated not by a quality threshold but by budget exhaustion: the supervisor's final decision was `stop` because the remaining budget (3,496 s) was smaller than the time the previous generation had taken to run (3,704 s). The system correctly identified that it could not complete another useful generation.

---

## 5. Discussion

### 5.1 When does tree persistence help?

Tree persistence adds the most value when three conditions hold simultaneously:
1. The benchmark measures a submitted artefact, not the agent's live behaviour.
2. The solution space is large enough that an improved scaffold can find genuinely new solutions, rather than re-converging to the same ones.
3. The meta-agent's scaffold improvements are incremental rather than discontinuous — a complete rewrite of the search strategy would invalidate the visit counts accumulated under the old strategy.

For agent artefact tasks, the argument is weaker. On LawBench, the "solutions" are the agent's classifications of individual instances, not standalone Python files. The tree structure does not carry across generations in a meaningful way because the per-instance state is implicit in the agent's prompt and weights, not in a persistent file.

### 5.2 Limitations.

**Visit count staleness.** When the meta-agent substantially changes the scaffold's search strategy, the visit counts from prior generations may mislead selection. A node that was heavily visited by a scaffold biased toward graph-diffusion approaches may be over-counted when the new scaffold pivots to matrix-factorisation approaches. Addressing this requires either a visit-count decay mechanism or per-strategy visit tracking, neither of which is currently implemented.

**Solution file bloat.** Without aggressive pruning, the `solutions/` directory can grow large on tasks with many cheap evaluations. The current implementation prunes to referenced nodes on copy-forward, but a more principled retention policy (e.g., keeping only Pareto-optimal solutions along multiple metric axes) could reduce storage pressure.

**Supervision calibration.** The supervision agent must balance exploration depth within a generation against the frequency of scaffold improvements. The current heuristic (evolve aggressively in early generations, gradually increasing generation length as the tree matures) is effective but not derived from first principles. A more formal model of when scaffold improvements are expected to unlock the most value — as a function of tree depth, cumulative node count, and per-generation marginal improvement — would make the supervision decision more robust.

---

## 6. Conclusion

The choice of self-improvement architecture should be conditioned on the nature of the final artefact. For agent artefact tasks, the scaffold is the deliverable and the standard SIA loop is the right tool. For solution artefact tasks, the scaffold is a search engine, and the tree of explored solutions is the primary asset. Discarding that tree at each generation boundary wastes the computation of every prior generation. Persistent PUCT state resolves this by carrying the full exploration history forward, allowing improved scaffolds to extend rather than restart the search. The architecture described here — generation-scoped solution storage, atomic tree updates, and a supervision loop sensitive to both tree depth and global budget — provides a concrete implementation of this idea. Preliminary results on the denoising task show that 254 solutions accumulated across six generations in a 12-hour run, with the largest single-generation improvement occurring in the final generation, when the tree was deepest and the scaffold most refined — precisely the regime where a restart from scratch would have been most costly.

---

## 7. Beyond Hill-Climbing for Agent-Artefact Tasks

The taxonomy in §2 distinguishes agent-artefact tasks from solution-artefact tasks, and §1 argues for tree search in the latter case. But the standard SIA loop also does pure hill-climbing in the former case: each generation always evolves from the *most recent* scaffold $A_g$, producing a single child $A_{g+1}$. The generational chain is linear by construction.

This is suboptimal for the same reason hill-climbing is generally suboptimal: a bad architectural decision made at generation $g$ propagates to all descendants. If $A_3$ restructures the scaffold around a retrieval strategy that turns out to be a dead end, $A_4$, $A_5$, and $A_6$ all start from that dead end. There is no mechanism to backtrack to $A_2$ and explore a different branch.

**The cost asymmetry.** Introducing tree search over scaffolds faces a practical obstacle that does not exist for solution-artefact tasks: evaluating a scaffold node requires running a full generation, which can cost from tens of minutes (for fast Q&A benchmarks like GPQA, with ~450 instances and simple API calls) to several hours (for multi-turn dialogue tasks or code execution benchmarks). Compare this to solution-artefact tasks where a node evaluation takes seconds. The effective tree depth reachable within a fixed budget is therefore much shallower for scaffold trees — on the order of 5–15 nodes rather than hundreds.

At such shallow depths, the overhead of a full PUCT selection policy is hard to justify. But the absence of full tree search does not imply that the linear chain is the only option. A minimal and immediately implementable improvement is to **always branch from the best-scoring scaffold** rather than the most recent one.

In the current SIA loop, if generation 3 scores 0.62 and generation 4 scores 0.58 (a regression), the feedback agent still receives $A_4$ as its input and tries to recover from it. Under the best-ancestor rule, the feedback agent would receive $A_3$ instead — the highest-scoring scaffold observed so far — and propose $A_4'$ as its child. This does not require tracking a branching tree, only a single pointer to the best generation. It costs nothing in terms of storage or evaluation overhead, and it prevents the compounding of accidental regressions.

A slightly stronger variant maintains a small explicit tree over scaffolds — say, depth 2 with branching factor 2 — and uses a simplified selection policy to decide which scaffold to extend next. Within a 10-hour budget, even on an expensive benchmark, this admits exploring 4–6 branches rather than a single linear chain, which is often enough to avoid the most common failure mode: committing irreversibly to one architectural direction after the first few generations.

The full unification — a single MCTS loop that searches jointly over solution space and scaffold space — remains an open design question. What this section argues is only that the current binary (full tree search for solution tasks, pure hill-climbing for agent tasks) understates the value of even modest structural improvements to the agent-task loop.

---

## Appendix: Symbol Table

| Symbol | Meaning |
|---|---|
| $g$ | Generation index |
| $A_g$ | Scaffold (target-agent package) at generation $g$ |
| $k$ | Supervision interval (nodes between supervision checks) |
| $c_\text{puct}$ | Exploration constant for the reference PUCT policy |
