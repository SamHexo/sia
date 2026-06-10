# SIA with Persistent Tree Search — or: why restarting from zero is a bad idea

Self-improving agents have a seductive loop: run → evaluate → rewrite → repeat. But most implementations share a hidden flaw — **each generation throws away everything the previous one learned about the search space**.

This is a write-up about a different design, one where the search tree survives across generations, and what happens when you actually run it.

---

## The problem with linear improvement

A typical SIA loop looks like this:

```
Gen 0 → target_agent_v0.py  →  score: 6.37×
           ↓ feedback
Gen 1 → target_agent_v1.py  →  score: 6.61×
           ↓ feedback
Gen 2 → target_agent_v2.py  →  score: ???
```

The meta-agent reads the feedback from gen N and rewrites the target agent for gen N+1. Clean, simple. But notice what's gone: every candidate solution gen N explored, every dead end it hit, every promising branch it started — all erased. Gen N+1 starts from a blank tree.

That means:
- **Duplicate work**: gen N+1 will re-explore paths gen N already tried
- **Lost diversity**: the new agent has no memory of which parts of the space are played out
- **Context collapse**: the meta-agent writes a new agent based on *one* feedback summary, not the actual geometry of the search space

---

## The persistent tree design

The core idea is simple: **the search tree is the persistent artifact, not the agent code**.

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
  ...
```

Each node in `tree_state.json` is a candidate solution with its score, parent, generation stamp, and PUCT statistics. Nodes are never deleted. New generations *continue* exploring the same tree — they can change their selection strategy, reweight exploration vs exploitation, add new heuristics — but the accumulated search history stays.

What the meta-agent now improves isn't "what code to try" but "**how to search**": the PUCT selection policy, the prompt given to the inner LLM, the stagnation recovery logic, the cross-pollination of top nodes into new prompts.

The **final artifact** is the best leaf node in the tree — the actual solution, not the scaffolding that found it.

---

## A supervision loop on top

One addition that pairs naturally with the persistent tree: a periodic supervisor that watches a running generation and decides whether to let it keep going or evolve the scaffold.

```
supervisor decision: CONTINUE | EVOLVE | STOP
```

The supervisor sees the tree state, the recent trajectory of scores, the bug rate, and makes a call. This is what lets the system exit early when a generation is stuck in a bad local region, rather than burning time hoping it recovers.

---

## Run 8 — GPU kernel optimization, 20 generations

To make this concrete: here's what one run looked like in practice, on a GPU kernel optimization task (AlphaFold-3 Outgoing TriMul, evaluated on H100). The target: maximize a speedup ratio versus a PyTorch reference kernel.

**Top-line numbers:**
- Starting score (gen 0): **6.37×**
- Best public score (gen 18): **7.37×** (+15.6%)
- Best private score (gen 18): **6.52×** (+37% vs gen 0 private)
- Total nodes evaluated across all generations: **431**
- Total wall time: ~7.6h

### How the tree grew across generations

The run splits naturally into 6 phases (families):

**F1 · Baseline (gen 0)** — First Triton kernel written from scratch. 5-step pipeline: fused LayerNorm FP16, projection + gating, BatchedGEMM, fused OutLayerNorm. Scores 6.37× public, 4.76× private. Budget of 300s hit after 1519s — the first run was exploratory.

**F2 · Infrastructure fixes (gen 1–5)** — The meta-agent spends 5 generations fixing the *scaffold*, not the kernel. Major bugs: the parent code was truncated to ~80 lines before being passed to the inner LLM (so it couldn't see the full kernel), UUID reuse causing node files to overwrite each other, PUCT Q-values not normalized (scores of ~6.6 were crushing the exploration term < 1.5, making selection fully greedy). After normalizing PUCT to [0,1] in gen 5, real exploration kicks in. Public score reaches 6.68×.

**F3 · Ghost generations (gen 6–9)** — A one-line indentation bug in `_trim_messages()` breaks the scope of `main()`. The target agent exits silently, writes no output, adds zero nodes to the tree. Four consecutive dead generations. The score is frozen at 6.68×. The tree is intact — nothing was lost — but four generation slots were wasted on a bug that kept reproducing.

The persistent tree matters here: when gen 10 finally fixes the bug, it immediately has access to all 431 nodes accumulated before gen 6. Nothing was thrown away.

**F4 · Task confusion (gen 9–10)** — Gen 9 hallucinates a task change ("Outgoing TriMul" → "TriMul") and adds a filter that hides all nodes from generations < 9. The 431-node tree is still there in the file — but the agent can't see it. Gen 10 fixes the indentation bug but the filter remains, causing a regression to 3.23×. A one-line filter masked the entire search history.

**F5 · Micro-optimizations (gen 11–13)** — Gen 11 removes the gen < 9 filter. The agent immediately sees the 6.68× node from gen 4 and uses it as a parent. In 3 submissions it reaches 6.77×. The key discovery in gen 13: **caching transposed FP16 weights in the `weights` dict during the warmup call** — eliminating repeated PyTorch allocations on every forward pass. First rule that changes code *structure* rather than kernel parameters. Public jumps to 7.05×, private to 6.07×.

**F6 · CUDA Graphs (gen 14–18)** — Gen 14 seeds a custom Triton GEMM benchmarked locally at 12.49×, but with a memory layout bug — the output `(B,N,N,H)` shape creates non-coalesced accesses downstream. Gen 15 reverts and introduces `torch.cuda.make_graphed_callables` via a `GraphWrapper`: captures the entire kernel sequence as a CUDA graph, eliminating ~0.15ms of CPU launch overhead per call. **Largest single jump of the run: +5.7%**, reaching 7.37× public. Gen 18 finalizes with BLOCK_K=32 and hardcoded `cols = tl.arange(0, 128)` to remove the D-loop. Gen 19 crashes on a duplicate keyword argument in the seed file — the run ends.

### Score trajectory

```
Gen   Public Best   Private   Notes
 0      6.37×       4.76×    Baseline kernel
 1      6.61×         —      Scaffold fixes
 2      3.25×       2.85×    Regression (0-byte parent)
 3      6.67×       4.59×    Selection fixed
 4      6.68×       4.76×    Cross-pollination added
 5      6.68×       4.73×    PUCT normalized
 6-9    6.68×         —      Ghost generations (indent bug)
10      3.23×       2.82×    Gen<9 filter still active
11      6.77×       5.08×    Filter removed, tree restored
12      6.79×       5.03×    Seeding 6.86×
13      7.05×       6.07×    Weight caching (+3.9%)
14      6.97×       5.98×    False lead (custom GEMM)
15      7.37×       6.50×    CUDA Graphs (+5.7%)
16      7.33×       6.43×    tl.constexpr
17      7.35×       6.49×    Reverted bad prompt
18      7.37×       6.52×    Best: BLOCK_K=32, hardcoded loop
19        —           —      Fatal crash
```

---

## What actually made the difference

Looking at the run honestly:

**The persistent tree was load-bearing in two specific moments.** Gen 11 — after four ghost generations and a task confusion — recovered immediately by accessing the 6.68× node from gen 4. Without persistence, that node would have been gone and the run would have started from whatever gen 11's meta-agent wrote fresh. Gen 13's weight caching discovery was built on that recovered node.

**Most generations were spent fixing scaffold bugs, not finding better kernels.** The meta-agent's main job turned out to be: fix the search infrastructure (UUIDs, file validation, PUCT normalization, bug detection), not invent new kernel architectures. The inner LLM, given a good enough scaffold, found the actual optimizations.

**The bug rate was consistently high (~70–80% of nodes).** This is a sign that the search space is hard and the inner LLM often generates syntactically correct but semantically broken kernels. The supervisor correctly flagged this as a reason to evolve — but the tree still captures the ~20–30% valid nodes and their scores.

**Ghost generations reveal a design tension.** When the target agent crashes silently, the supervisor detects the abnormal exit and triggers the next generation. But the meta-agent sees a generation that "ran" with no output — it can misattribute the failure. Better crash instrumentation (structured error types written before exit) would help here.

---

## The artifact

The target agent that gets handed off at the end of a run is not `gen_18/target_agent/target_agent.py` — that's the scaffold. The artifact is the best node in `tree_state.json`: a standalone kernel file that can be dropped directly into the evaluation harness.

The scaffold was how you got there. The kernel is what you keep.

---

## What's next

A few open questions this run surfaces:

- **Smarter supervision**: the current supervisor triggers on bug rate and node count. A richer signal — trajectory curvature, diversity of recently explored nodes, estimated remaining compute — would make EVOLVE decisions less reactive.
- **Cross-run tree transfer**: if two runs on related tasks share a tree format, can gen 0 of run 2 seed from the best nodes of run 1?
- **Meta-agent access to the full tree**: the meta-agent currently sees a text summary of the tree. Giving it direct structured access (top-N nodes, score distributions by branch) might reduce hallucinations like the gen 9 task confusion.
- **Ghost generation detection**: the supervisor can detect a silent exit, but the *next* meta-agent needs to know it was a crash, not a normal run. Encoding failure type in the tree node would close this loop.
