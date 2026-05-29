## Scaffold Design Philosophy

The target agent is a **search algorithm**, not a chatbot. Its job is to explore a solution
space systematically and find the best possible solution within a time budget. The meta-agent
has complete creative freedom over how that search is structured.

---

### Architecture: double loop

The reference implementation uses a clean two-level structure — adopt it or improve on it:

```
outer loop  — MCTS / search strategy
              → select a node to expand (PUCT or any policy)
              → allocate a candidate path
              → run inner agent
              → evaluate (scaffold runs evaluate.py directly)
              → write node to state.json
              → repeat

inner loop  — short multi-turn LLM (focused on ONE task)
              → receives: parent solution + score + task context
              → goal: write ONE solution file and stop
              → can explore dataset, read existing solutions, debug in bash
              → stops when file is written or max_inner_turns reached
```

**Key principle**: the scaffold controls all search decisions. The inner LLM is a
*proposal generator*, not a strategist. It writes code — it does not decide which
branch to explore, which family to abandon, or when to restart.

---

### Search strategy — be creative and task-specific

The reference agent implements basic PUCT. You are free to go much further. Think about
what actually makes sense for the task at hand, not generic templates.

**Family / branch management**
- Track which nodes belong to the same "strategy family" (e.g. magic-based, knn-based,
  autoencoder-based) using node metadata in state.json
- Mark a family as exhausted if its last K expansions showed no improvement
- Stop expanding exhausted families and shift budget toward unexplored ones
- This prevents the search from getting stuck in a local optimum

**Restarts from scratch**
- If the overall best score hasn't improved in N iterations, generate a completely new
  candidate from scratch (no parent) to escape local optima
- Use a diversity-forcing prompt for restart nodes: "try an approach fundamentally
  different from all existing solutions"

**Fusion / crossover**
- Select two high-scoring nodes from different families and ask the LLM to combine their
  approaches: "Solution A is strong on X, Solution B is strong on Y — produce a hybrid"
- Inject both parent solutions into the inner agent's prompt

**Beam search variant**
- Instead of one expansion per iteration, run K inner agents in parallel (subprocess or
  concurrent.futures) and keep only the top-K results
- Use cheap sampling (high temperature) to generate diverse candidates, expensive model to refine

**Simulated annealing / temperature scheduling**
- Start with high `c_puct` (exploration) early in the run, decay toward exploitation
- Track elapsed time and dynamically adjust the balance

**Population-based search**
- Maintain a population of N "active solutions"
- Each iteration: sample a pair, recombine, evaluate, replace worst if better

---

### What to put in the prompt to the inner LLM

The inner LLM works best when the scaffold prepares its context:

- **Inject the parent solution** with its score — do not make the LLM figure out what to build on
- **Inject the best solution so far** (different family) — useful for fusion rounds
- **Inject dataset statistics** if the scaffold read them: shape, sparsity, baseline metrics
- **Inject explicit strategy directives**: "tune knn and t only", "try a completely different
  denoising method", "fix this specific error: {error}"
- **Show top-3 solutions from state.json** for context on what has and hasn't worked
- **Be specific**: "increase knn from 5 to 10–20 and t from 3 to 5–15" beats "try different parameters"

The inner LLM should receive **structured, informed context** — not a blank task description.

---

### Minimal LLM delegation — scaffold does the heavy lifting

**The scaffold should handle** (Python, not LLM):
- Read and summarize state.json to select parents, detect stagnation, track families
- Run evaluate.py and parse RESULT_JSON — the LLM never sees raw scores from evaluate
- Detect exhausted branches, trigger restarts, decide fusion rounds
- Read dataset files and compute basic statistics to inject into prompts
- Implement all retry/fallback logic on crashes or missing output

**The inner LLM should handle** (one-shot code generation):
- Write a `custom_denoise` (or equivalent task function) based on the provided context
- Optionally: explore dataset structure or existing solutions in bash before writing
- Optionally: run a quick sanity check (import test) before committing to the final file

**The inner LLM should NOT**:
- Decide which node to expand (that's PUCT / scaffold)
- Call evaluate.py — the scaffold does this
- Make search strategy decisions — just write good code given the context

---

### state.json as shared memory

The target agent has full read/write access to state.json. Use it:

```python
from utils.tree_utils import read_state, get_best_node, get_generation_nodes

tree = read_state(args.state_file)
nodes = tree["nodes"]

# Find all nodes with a specific tag
family_knn = [n for n in nodes.values() if n.get("metadata", {}).get("family") == "knn"]

# Add metadata when writing a node
result["metadata"] = {"family": "knn", "knn_value": 10}
write_tree_node(state_file, path, result, gen)  # pass metadata via result dict
```

You can extend `write_tree_node` or add helpers in `utils/` to track anything — strategy
tags, parent chains, error logs, anything that helps the search strategy.

---

### Domain knowledge — use it concretely

Do not write a generic scaffold. The meta-agent knows the task. Use that knowledge:

- If the task is **scRNA-seq denoising**: you know MAGIC, ALRA, DCA, scVI are relevant.
  Pre-populate the first few expansions with these specific baselines rather than asking
  the LLM to "try different methods".
- If the task involves **hyperparameter tuning**: implement a structured grid or Bayesian
  approach rather than random guessing.
- If the task has a **known score ceiling** (from task.md or baselines): use it to decide
  when to stop exploitation and switch to new approaches.

Concrete domain knowledge in the scaffold always beats "ask the LLM to figure it out".

---

### utils/ — helpers you can use or extend

```python
from utils.tree_utils import write_tree_node, read_state, get_best_node, get_generation_nodes, summarize_tree
```

- `write_tree_node(state_file, solution_path, result, generation)` — registers a node;
  if solution_path is already in solutions/, skips the copy (UUID taken from filename)
- `read_state(state_file)` — raw tree dict
- `get_best_node(state_file)` → `(node, score)`
- `get_generation_nodes(state_file, generation)` → list of nodes from one gen
- `summarize_tree(state_file)` → compact text summary (for supervision calls)

Add any reusable components to `utils/` — family trackers, prompt builders, search policies.

---

### Supervision — no fixed turn limit

The outer loop runs until the supervision system stops it. Do NOT implement your own
hard timeout. The supervision model checks every `--supervision_interval` MCTS iterations
and returns CONTINUE / EVOLVE / STOP.

Design your outer loop to be responsive to supervision: check after each node expansion
(not only after long batches), so the signal is acted on quickly.
