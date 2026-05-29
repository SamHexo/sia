"""
Per-iteration supervision call for target agents.

Import via sys.path after adding --shared_dir:
    from supervision_call import check_supervision

Returns a dict {"decision": str, "reason": str} where decision is one of:
- "continue"  — keep running, progress is being made
- "evolve"    — stop this generation cleanly, meta-agent will create a new scaffold
- "stop"      — terminate the entire run (goal met or budget exhausted)
- "fail"      — supervision LLM call failed (caller decides how to handle)

reason is a 1-2 sentence explanation of the decision.

When called from the orchestrator, pass prompt_path pointing to
orchestration/prompts/supervision_prompt.md (which also accepts {LATEST_LOGS}).
When called from a target agent, prompt_path is omitted and the built-in prompt is used.
"""

import re as _re

_PROMPT = """\
You are supervising a running AI search. Decide whether it should keep going.

Generation: {CURRENT_GEN} | Elapsed (this gen): {ELAPSED}s | Budget remaining: {REMAINING}s

{TREE_SUMMARY}

Rules:
- CONTINUE  if scores are still improving or stagnation is low and budget allows
- EVOLVE    if the search is stuck, cycling, or a structural scaffold change would help
- STOP      if the best score meets the task target, budget is nearly exhausted, or a fatal error exists

Reply with exactly one word on the first line (CONTINUE, EVOLVE, or STOP), then 1–2 sentences explaining why.
"""


def _fill(template: str, **kw: str) -> str:
    return _re.sub(r"\{([A-Z_][A-Z0-9_]*)\}", lambda m: kw.get(m.group(1), m.group(0)), template)


def check_supervision(
    tree_summary: str,
    elapsed_seconds: float,
    remaining_seconds: float,
    current_gen: int,
    model: str,
    prompt_path: str | None = None,
    latest_logs: str = "",
    gen0_evolve_duration_s: int = 0,
) -> dict:
    """Make a cheap single-turn LLM call to decide CONTINUE / EVOLVE / STOP.

    Returns {"decision": str, "reason": str}.

    Args:
        tree_summary:     Compact text summary of current search state.
        elapsed_seconds:  Seconds elapsed since the generation started.
        remaining_seconds: Seconds left in the safety timeout.
        current_gen:      Current generation number (for context).
        model:            Full litellm model string (e.g. "gemini/gemini-3.1-pro-preview").
        prompt_path:      Optional path to a .md template file (e.g. supervision_prompt.md).
                          If omitted, the built-in _PROMPT above is used.
        latest_logs:      Optional recent stdout tail — used when prompt_path is provided.
    """
    import time as _time
    import logging as _logging
    _log = _logging.getLogger(__name__)

    # Gen-0 fast path: no LLM call — purely time-based EVOLVE signal
    if current_gen == 0 and gen0_evolve_duration_s > 0:
        if elapsed_seconds >= gen0_evolve_duration_s:
            reason = f"Gen-0 time budget reached ({elapsed_seconds:.0f}s >= {gen0_evolve_duration_s}s)."
            _log.info(f"Gen-0 evolve threshold reached → evolve")
            return {"decision": "evolve", "reason": reason}
        return {"decision": "continue", "reason": "Gen-0 still within time budget."}

    template = open(prompt_path, encoding="utf-8").read() if prompt_path else _PROMPT
    prompt = _fill(
        template,
        CURRENT_GEN=str(current_gen),
        ELAPSED=f"{elapsed_seconds:.0f}",
        REMAINING=f"{remaining_seconds:.0f}",
        TREE_SUMMARY=tree_summary,
        LATEST_LOGS=latest_logs,
    )

    delays = [0, 5, 10, 20]
    last_exc = None
    for attempt, delay in enumerate(delays):
        if delay > 0:
            _time.sleep(delay)
        try:
            import litellm
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            # First line = decision word; remaining lines = reason
            lines = text.splitlines()
            first = lines[0].strip().upper() if lines else ""
            reason = " ".join(l.strip() for l in lines[1:] if l.strip()) or first
            if "STOP"   in first: return {"decision": "stop",    "reason": reason}
            if "EVOLVE" in first: return {"decision": "evolve",  "reason": reason}
            return {"decision": "continue", "reason": reason}
        except Exception as _e:
            last_exc = _e
            _log.warning(f"Supervision attempt {attempt + 1}/{len(delays)} failed: {_e}")

    reason = f"All {len(delays)} supervision attempts failed — last error: {last_exc}"
    _log.error(reason)
    return {"decision": "fail", "reason": reason}
