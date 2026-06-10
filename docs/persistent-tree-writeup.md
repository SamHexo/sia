# SIA with Persistent Tree Search — the right artifact determines the right design

Self-improving agents share a common loop: run → evaluate → rewrite → repeat. Simple and powerful. But before you build one, there's a question worth spending time on: **what is the artifact you actually want at the end?**

The answer changes everything about how you should design the system.

---

## Two modes, two designs

### Mode 1: the artifact is the agent

Imagine you're building an agent that reviews ML papers, or one that solves agentic coding tasks. You run it on a benchmark, get a score, improve it, repeat. After 20 generations, what you ship is the **agent itself** — a general-purpose system you'll reuse on similar tasks, deploy in a product, or hand to a team.

In this mode, **restarting from zero each generation makes perfect sense**. The exploration state from gen N is irrelevant — what matters is the agent's *code*, its prompts, its reasoning patterns. The meta-agent synthesizes feedback and produces a better agent. Each generation is a clean rewrite. Information transferred across generations should be minimal: ideally just a performance summary so the meta-agent knows whether it's improving. Carrying over the search tree would actually be noise — the tree is full of task-specific solutions that don't generalize.

### Mode 2: the artifact is the solution

Now imagine the goal is different: you want the **best possible solution to a specific, well-defined task**. A GPU kernel for a particular operation. A denoising algorithm on a fixed dataset. A strategy for a specific benchmark. You'll run this once, evaluate it, and submit. The final deliverable isn't the agent — it's what the agent *found*.

In this mode, **restarting from zero is actively harmful**. Every generation that starts fresh throws away explored territory. The next generation will rediscover paths the previous one already visited, re-evaluate solutions that were already found to be mediocre, and burn compute re-learning the shape of a search space that was already partially mapped.

Even restarting from the *best solution found so far* is problematic. You keep the local peak, but you lose all the branches that were promising-but-not-best, all the regions that turned out to be dead ends (so you don't explore them again), and all the variance information that tells you where the unexplored territory actually is. You're climbing from the summit of the last attempt with no memory of the mountain.

### Nodes evaluated is the key metric — not scaffold quality

There's a subtler implication that's easy to miss. Suppose you have a 10-hour compute budget and you're choosing between two strategies:

- **Strategy A**: Run 10 independent 1-hour generations, each restarting from zero. Each generation has a highly optimized scaffold — clean prompts, smart heuristics, well-tuned LLM calls. By gen 10, the scaffold is excellent.
- **Strategy B**: Run 10 hours with a persistent tree and a scaffold that starts mediocre. Each generation is given a chance to improve the search strategy, but the tree accumulates continuously.

Strategy A sounds better. But consider what actually happens: each restart-from-zero generation spends its first N evaluations re-discovering the shape of the space, re-finding the good region, re-establishing a baseline. The total number of *distinct, non-redundant* nodes evaluated across 10 generations might be far less than it appears — a lot of compute goes into rediscovering what was already known.

Strategy B's tree, even with an imperfect scaffold, keeps growing. Every node is new territory. By hour 10, the tree might have explored 3–5× more distinct regions of the solution space, even if no single generation was as polished as Strategy A's gen 10.

**The intuition**: in a fixed compute budget, a slightly worse harness that explores 1000 nodes usually beats a better harness that explores 200. The number of evaluations compounds — a good node found in hour 2 becomes a parent that enables better mutations in hours 3–10. With restarts, that compounding resets each generation.

This doesn't mean the scaffold quality is irrelevant. A scaffold that generates 80% broken nodes is burning most of its evaluations. But it means that *between a faster-iterating mediocre scaffold and a slower, more careful scaffold*, the fast one is often better — as long as the tree persists and compounding can happen.

---

## What persistent tree search actually means

The idea is to keep the search state — the tree — as the **primary persistent artifact** across all generations. Not the solution, not the agent code: the tree.

```
runs/run_001/
  research_state/
    tree_state.json        ← survives ALL generations
  gen_0/
    target_agent/
      target_agent.py      ← scaffold v0, reads+writes the tree
      conf.yaml
  gen_1/
    target_agent/
      target_agent.py      ← scaffold v1, continues from the same tree
      conf.yaml
  gen_2/
    target_agent/
      target_agent.py      ← scaffold v2, same tree, now with 200+ nodes
      conf.yaml
```

Every candidate solution ever evaluated lives in `tree_state.json` as a node: its code, its score, its parent, which generation created it, and its PUCT visit statistics. Nodes are never deleted.

What changes across generations is the **scaffold** — the target agent's search strategy. The meta-agent can change how PUCT selects the next node to expand, how the inner LLM is prompted to generate mutations, when to restart from a different branch, how to cross-pollinate ideas from the top-scoring nodes. What it *cannot* do (by design) is erase the tree. The search history is fixed; only the search strategy evolves.

The delivered artifact at the end is the **best leaf node** — extracted from the tree and handed off as a standalone solution.

```
tree_state.json  →  best_node.code  →  solution.py  ✓
```

The scaffold that found it is discarded. You don't need it anymore.

---

## The mechanisms that make it work

A persistent tree alone isn't enough. Several mechanisms were built on top to make generation-over-generation improvement actually happen:

### PUCT search with normalized Q-values

The tree uses PUCT (Predictor + Upper Confidence bound for Trees) to balance exploitation of known-good nodes and exploration of new branches. One critical detail: Q-values must be normalized to [0,1] before computing the PUCT score.

This sounds obvious but it bit us hard in early runs. Raw scores of ~6.6× completely swamped the exploration term (< 1.5), making selection purely greedy — the agent would hammer the same best node over and over rather than branching out. Once Q-values were normalized, real exploration kicked in.

### Cross-pollination

When the inner LLM is asked to generate a mutation of a parent node, it also receives the code of the **top-2 scoring nodes in the entire tree** (excluding the selected parent) injected into its system prompt. This lets it merge patterns from different branches — taking a memory optimization from one lineage and a kernel structure from another — without the PUCT selection needing to explicitly plan that.

### Stagnation recovery

If no new best score is found after N iterations, the agent forces selection away from the current local region. It picks a node from a different subtree as the new parent and injects a "fresh start" prompt that includes top context from the best nodes so far. This prevents the agent from getting trapped grinding one peak.

### Seeding

Rather than always starting gen 0 from a blank tree, a known good solution can be evaluated and inserted as the root node. Subsequent generations see this seed as their starting point. In run 8, updating the seed from 6.37× to 6.86× to 7.04× to 7.40× across generations gave the inner LLM a progressively better baseline to mutate from.

### Supervision loop

This is the piece that connects everything. The orchestrator runs a **supervision agent** on a timer that periodically queries an LLM to evaluate the current generation's trajectory and issue one of three decisions:

```
CONTINUE  — the generation is making progress, let it run
EVOLVE    — trigger the meta-agent now, start the next generation
STOP      — the run is done, extract the best node
```

The supervisor sees: the current tree state summary, the score trajectory of the last N nodes added, the bug rate (ratio of invalid submissions), and the generation's elapsed time versus budget.

This matters because **generations don't have a natural stopping point**. A generation could find a good node in its first three submissions and then plateau for an hour. Or it could be stuck in a broken region but one more branch might unlock a jump. The supervisor makes this call dynamically rather than running each generation to a fixed time limit.

In practice, the supervisor's most reliable signal was the bug rate. When 70–80% of newly evaluated nodes are syntactically broken kernels, the current scaffold is generating in a region of the code space where the inner LLM keeps failing. That's a strong signal to evolve — not because the tree is bad, but because the *strategy* needs to change.

---

## Run 8 — GPU kernel optimization, 20 generations

One run on a GPU kernel optimization task (AlphaFold-3 Outgoing TriMul, T4). Goal: maximize speedup ratio vs. a PyTorch reference. Numbers below aren't meant to be compared to other hardware — the T4 is what it is — but the structure of how the score evolved is what's interesting.

**Top-line:**
- 431 nodes evaluated across 20 generations, ~7.6h wall time
- Gen 0 → Gen 18: **+15.6%** on public score, **+37%** on private score
- 6 distinct evolution families in the tree

The run splits into those 6 families, each corresponding to a different meta-agent strategy:

**F1 · Baseline** — First Triton kernel written from scratch. 6 nodes, establishes the starting score.

**F2 · Infrastructure fixes (5 gens)** — The meta-agent spends five generations fixing the *scaffold*, not the kernel: parent code was truncated to ~80 lines before being passed to the inner LLM, UUID reuse caused node files to overwrite each other, and un-normalized PUCT (raw scores of ~6.6× crushing the exploration term < 1.5) made selection fully greedy. Once fixed, real exploration starts. **+4.8%** over baseline.

**F3 · Ghost generations (4 gens, 0 nodes)** — A one-line indentation bug in `_trim_messages()` breaks the scope of `main()`. The target agent exits silently. Four consecutive generations add zero nodes. The supervisor detects the abnormal exit each time (no `exit_reason.txt` written) and triggers the next generation. The tree is intact — all prior nodes preserved — but four generation slots burned on a reproducible crash.

**F4 · Task confusion** — The meta-agent hallucinates a task change and adds a filter hiding all nodes from generations < 9. The full tree is still in the file; the agent just can't see it. Causes a regression to near-zero before the next generation partially recovers.

**The tree mattered at this junction**: once the filter was removed in F5, the agent immediately found the best node from F2 (four families back) and used it as a parent. Three submissions later: new best. Starting fresh would have meant that node was gone.

**F5 · Micro-optimizations** — Full tree visible again. Key discovery: **cache transposed FP16 weights in the `weights` dict during warmup**, eliminating repeated PyTorch allocations on every forward pass. First change that touches code *structure* rather than kernel parameters. **+3.9%** in one generation.

**F6 · CUDA Graphs (5 gens)** — Introduction of `torch.cuda.make_graphed_callables` via a `GraphWrapper`: captures the entire kernel call sequence as a CUDA graph, eliminating CPU launch overhead per forward pass. Largest single jump: **+5.7%**. Final generations stabilize with BLOCK_K=32 tuning. Run ends on gen 19 with a fatal crash (duplicate keyword argument in the seed file).

---

## Honest takeaways

**The persistent tree was load-bearing twice.** Gen 11's immediate recovery after four ghost generations and a task confusion, and gen 13's weight caching discovery built on top of that recovered state. In a restart-from-zero design, both of those inflection points would have been reset.

**Most generations were spent fixing scaffold bugs.** The meta-agent's real job was: fix the infrastructure (UUIDs, file validation, PUCT normalization, crash detection), not invent new kernel architectures. The inner LLM, given a working scaffold with good context, found the actual optimizations.

**The bug rate (~70–80%) was the supervisor's most reliable signal.** When the scaffold is generating into a region where the inner LLM keeps producing broken kernels, evolving the strategy makes more sense than waiting. The supervisor caught this repeatedly.

**Ghost generation detection is an open problem.** The supervisor correctly identified the silent exits, but the meta-agent's input was a generation that "ran" with no output — making it hard to distinguish a crash from a slow run. Structured error types written to the tree node before any exit would close this loop.

**The design only makes sense when the artifact is the solution.** If the goal were to build a reusable kernel-writing agent for a product, restarting from zero each generation would be the right call — the scaffold would generalize, the search history wouldn't. The persistent tree is specifically for the case where you want the best answer to this precise task, and losing exploration history is a real cost you want to avoid.

---

## Open question: does this transfer to different benchmarks?

GPU kernel optimization is a friendly setting for persistent tree search: the score is deterministic (run the kernel, measure speedup), the search space is continuous enough that neighboring nodes tend to have related scores, and the "artifact" is unambiguous (a piece of code that either runs fast or doesn't).

It's worth asking whether this holds for benchmarks where the evaluation is fundamentally different.

Take GPQA — a graduate-level science reasoning benchmark. Each question has one correct answer. The score of a "solution" (a reasoning strategy, a prompt, a chain-of-thought scaffold) is measured over a distribution of questions, not a single deterministic call. Two runs of the same strategy on the same question can give different answers. The search space isn't a smooth landscape of kernel parameters — it's a much noisier, higher-dimensional space of reasoning behaviors.

A few things that might shift:

**The Q-value signal is noisier.** In GPU kernels, a node's score is exact. In GPQA, a strategy's score is an average over many evaluations, and a single evaluation is stochastic. PUCT selection depends on reliable Q-values. With high-variance scores, the tree might spend many nodes disambiguating between genuinely different strategies vs. noise in the evaluation.

**Dead ends are less informative.** In code optimization, knowing "this branch explored heavy Triton GEMM customization and it didn't work" is useful — you can avoid that direction. In reasoning benchmarks, knowing "this chain-of-thought style underperformed" is fuzzier — it might have underperformed *on these questions*, or *with this model*, or just due to sampling variance. The tree retains the information, but it's harder to act on.

**The solution itself is less separable from the scaffold.** A GPU kernel can be extracted from the tree and run standalone. A reasoning strategy is more entangled with the agent that applies it — you'd be extracting a prompt or a behavior, not a file. Whether the "best node" in the tree is truly portable is a more open question.

That said, the core argument still holds: if you're optimizing for a *specific* benchmark and the artifact is the best score you can achieve on it, you don't want to lose exploration history between generations. The question is whether the tree format — designed around deterministic, code-level scores — needs to adapt for stochastic, behavior-level evaluations. Maybe nodes need confidence intervals instead of point scores. Maybe selection should weight uncertainty differently. Maybe generations need to explicitly re-evaluate promising nodes to reduce noise before building on them.

It's an open question whether the gains from persistent search in a setting like GPQA would outweigh the added complexity of managing a noisy tree — or whether a cleaner separation (persistent *memory* of what's been tried, but no explicit PUCT over it) would be the better fit there.
