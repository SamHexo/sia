# SIA — persistent tree search

The core question before building a self-improving agent: **what do you want at the end?**

---

## Two cases

**Case 1: you want the agent.** You're building something reusable — a coding agent, a research assistant. After N generations, you ship the agent itself. Here, restarting from zero each generation makes sense. The search state is task-specific noise; only the agent code matters. Pass a performance summary to the meta-agent, that's it.

**Case 2: you want the solution.** You have a specific task — a GPU kernel, a denoising algorithm, a benchmark. The deliverable is the best result you can produce on that exact task. Here, restarting from zero is wasteful: every generation re-explores territory the previous one already mapped, re-evaluates dead ends, rediscovers the good region from scratch. Even restarting from the best solution found so far is problematic — you keep the local peak but lose all the branching information around it.

### The compute budget argument

For a fixed budget — say 10 hours — consider two options:

- 10 generations × 1h each, restarting from zero. By gen 10, the scaffold is good. But it only runs for 1 hour.
- 1 generation × 10h with a persistent tree. The scaffold never improves. But the tree keeps growing for 10 hours.

The second often wins, because a good node found in hour 2 becomes a parent for better mutations in hours 3–10. Restarts throw that compounding away.

The real sweet spot: spend a few short generations improving the scaffold, then let a good generation run long. 9 gens × 1h + 1 gen × 10h will likely beat both extremes — and the persistent tree is what makes it possible, because the early short generations aren't wasted.

---

## How it works

The tree — `tree_state.json` — is the persistent artifact. It holds every candidate solution ever evaluated: code, score, parent, generation, PUCT stats. It never gets reset.

```
runs/run_001/
  research_state/
    tree_state.json     ← survives all generations
  gen_0/
    target_agent/
      target_agent.py
      conf.yaml
  gen_1/
    target_agent/
      target_agent.py   ← different scaffold, same tree
      conf.yaml
```

The meta-agent rewrites the scaffold each generation — how the tree is traversed, how the inner LLM is prompted, what heuristics are used. The only constraint: it reads and writes `tree_state.json`. It cannot erase it.

At the end, the artifact is the best node extracted from the tree — a standalone solution. The scaffold that found it gets discarded.

---

## Two things built on top

**Gen 0 — plain PUCT.** The first generation is a classic PUCT search: nodes have visit counts and Q-values, selection balances exploitation vs. exploration. It runs for a fixed time budget, no supervision. Just builds a baseline tree.

**Supervision loop (gen 1+).** From gen 1 onwards, a supervision agent runs on a timer. It reads the current tree state and recent trajectory, and sends a guidance signal to the running generation — is this progressing or stagnating? The generation uses that signal however its scaffold sees fit. This is also how the system decides when to trigger the next generation, without a fixed clock.

**Target agent is a folder.** The agent isn't a single file:

```
gen_N/target_agent/
  target_agent.py
  conf.yaml
  utils/
    tree_utils.py
    ...
```

The meta-agent can add files, restructure, add utility modules. Everything is fair game as long as the tree interface is respected.

---

## Run 8 — GPU kernel, 20 generations (T4)

Task: maximize speedup ratio vs. PyTorch reference for AlphaFold-3 Outgoing TriMul.

- 431 nodes across 20 generations, ~7.6h
- Gen 0 → Gen 18: **+15.6%** public, **+37%** private
- 6 evolution families

**F1 · Baseline** — First Triton kernel, 6 nodes.

**F2 · Infra fixes (5 gens)** — Meta-agent spends 5 gens fixing the scaffold: truncated parent code, UUID reuse overwriting node files, un-normalized PUCT making selection fully greedy. Once fixed: **+4.8%**.

**F3 · Ghost generations (4 gens, 0 nodes)** — One-line indentation bug breaks `main()` scope. Agent exits silently, writes nothing. Supervisor detects the abnormal exit and triggers next generation each time. Tree intact, 4 generation slots wasted.

**F4 · Task confusion** — Meta-agent hallucinates a task change, adds a filter hiding all nodes from gen < 9. The tree is still there, the agent just can't see it. Regression to near-zero.

**Where the tree mattered:** once the filter was removed, the agent immediately picked up the best node from F2 — four families back. Three submissions later: new best. With restarts, that node would have been gone.

**F5 · Micro-opts** — Tree visible again. Key find: cache transposed FP16 weights during warmup, eliminating repeated allocations. **+3.9%**.

**F6 · CUDA Graphs (5 gens)** — `torch.cuda.make_graphed_callables` captures the kernel sequence as a CUDA graph, removing CPU launch overhead. Biggest jump: **+5.7%**. Run ends at gen 19 on a fatal crash (duplicate keyword argument in seed).

---

## Open question: other benchmarks

GPU kernels are a good fit: scores are deterministic, the search space is smooth, the artifact is a standalone file.

GPQA-style benchmarks are harder. Scores are stochastic (averaged over many questions), dead ends are noisier (did this strategy fail, or was it just bad luck on these questions?), and the "solution" is a behavior entangled with the agent, not a separable file. The tree still makes sense in principle, but Q-values would need confidence intervals, and selection would need to account for noise. Whether that's worth the complexity over just keeping a persistent *memory* of what's been tried — without explicit PUCT — is an open question.
