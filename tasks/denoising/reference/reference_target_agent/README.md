# Target Agent

MCTS outer loop + focused inner multi-turn LLM per node.

## Architecture

| Layer | Description |
|-------|-------------|
| Outer | PUCT tree search — selects parent node, runs inner agent, logs results |
| Inner | Multi-turn LLM that calls `submit_solution(code)` to evaluate and register solutions |
| Supervision | Every `supervision_interval` iterations, asks supervision model to CONTINUE / EVOLVE / STOP |

## Files

| File | Purpose |
|------|---------|
| `target_agent.py` | Main agent |
| `conf.yaml` | PUCT and search parameters |
| `utils/tree_utils.py` | State tree helpers |

## Packages

| Package | Purpose |
|---------|---------|
| `pyyaml` | Load conf.yaml |

## Changelog

| Generation | Change |
|------------|--------|
| ref | Initial scaffold — MCTS + inner multi-turn LLM with submit_solution |
