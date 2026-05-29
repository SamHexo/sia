#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID=1
EXP_DURATION_MIN=120              # total experiment budget; safety timeout = 1.3×
GEN0_EVOLVE_DURATION_MIN=5        # gen-0 runs for this long before auto-EVOLVE (no LLM call)
META_AGENT_MAX_TURNS=100          # turn budget for the meta-agent's own tool loop
TASK_MODEL_TEMPERATURE=0.3
META_MODEL="gemini/gemini-3.1-pro-preview"
SUPERVISION_MODEL="gemini/gemini-3.1-pro-preview"
SUPERVISION_INTERVAL=5
SUPERVISION_CHECK_TIMEOUT_MIN=10
TASK_MODEL="tinker://55bc74de-c858-54ce-9756-e6f54d7a5a8d:train:0/sampler_weights/000049"
BACKEND="openhands"
# Seed the gen-0 tree with the FP16 reference solution before the MCTS loop.
SEED_SOLUTION="${SCRIPT_DIR}/reference/reference_solution.py"

# ── CLI args ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_id)                        RUN_ID="$2";                        shift 2 ;;
    --exp_duration_min)              EXP_DURATION_MIN="$2";              shift 2 ;;
    --gen0_evolve_duration_min)      GEN0_EVOLVE_DURATION_MIN="$2";      shift 2 ;;
    --task_model_temperature)        TASK_MODEL_TEMPERATURE="$2";        shift 2 ;;
    --meta_model)                    META_MODEL="$2";                    shift 2 ;;
    --task_model)                    TASK_MODEL="$2";                    shift 2 ;;
    --backend)                       BACKEND="$2";                       shift 2 ;;
    --supervision_model)             SUPERVISION_MODEL="$2";             shift 2 ;;
    --supervision_interval)          SUPERVISION_INTERVAL="$2";          shift 2 ;;
    --supervision_check_timeout_min) SUPERVISION_CHECK_TIMEOUT_MIN="$2"; shift 2 ;;
    --meta_agent_max_turns)          META_AGENT_MAX_TURNS="$2";          shift 2 ;;
    --seed_solution)                 SEED_SOLUTION="$2";                 shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
# ─────────────────────────────────────────────────────────────────────────────

YELLOW="\033[1;33m"; NC="\033[0m"
printf "${YELLOW}[gpu-kernel]${NC} A CUDA GPU is required for this task.\n"
printf "${YELLOW}[gpu-kernel]${NC} Make sure torch and triton are installed in the run venv.\n\n"

python orchestration/orchestrator.py \
  --task_dir                       ./tasks/gpu-kernel \
  --exp_duration_min               "${EXP_DURATION_MIN}" \
  --gen0_evolve_duration           "$((GEN0_EVOLVE_DURATION_MIN * 60))" \
  --task_model_temperature         "${TASK_MODEL_TEMPERATURE}" \
  --run_id                         "${RUN_ID}" \
  --backend                        "${BACKEND}" \
  --meta_model                     "${META_MODEL}" \
  --task_model                     "${TASK_MODEL}" \
  --supervision_model              "${SUPERVISION_MODEL}" \
  --supervision_interval           "${SUPERVISION_INTERVAL}" \
  --supervision_check_timeout_min  "${SUPERVISION_CHECK_TIMEOUT_MIN}" \
  --meta_agent_max_turns           "${META_AGENT_MAX_TURNS}" \
  ${SEED_SOLUTION:+--seed_solution "${SEED_SOLUTION}"}
