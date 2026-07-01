#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── LiteLLM proxy ─────────────────────────────────────────────────────────────
# LITELLM_API_KEY must be set in your environment (e.g. in local/secrets.sh).
# LITELLM_BASE_URL is exported here so all subprocesses pick it up automatically.
export LITELLM_BASE_URL="https://litellm-837479182951.us-west2.run.app"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID=1
EXP_DURATION_MIN=720              # total experiment budget; safety timeout = 1.3×
GEN0_EVOLVE_DURATION_MIN=5      # gen-0 runs for this long before auto-EVOLVE (no LLM call)
META_AGENT_MAX_TURNS=100         # turn budget for the meta-agent's own tool loop
TASK_MODEL_TEMPERATURE=0.3
SUPERVISION_INTERVAL=5
SUPERVISION_CHECK_TIMEOUT_MIN=20

# ── Model preset — pick ONE block and comment out the others ──────────────────

# Preset A: GPT-5.5 via LiteLLM proxy (meta + target)
#META_MODEL="azure_ai/gpt-5.5"
#TASK_MODEL="azure_ai/gpt-5.5"
#SUPERVISION_MODEL="azure_ai/gpt-5.5"
#BACKEND="openhands"

# Preset B: Claude via LiteLLM proxy (meta + target)
# META_MODEL="azure_ai/claude-opus-4-8"
# TASK_MODEL="azure_ai/claude-opus-4-8"
# SUPERVISION_MODEL="azure_ai/claude-opus-4-8"
# BACKEND="openhands"

# Preset C: Claude native SDK (meta) + GPT-5.5 proxy (target)
# META_MODEL="claude-sonnet-4-6"     # Claude Code SDK — uses ANTHROPIC_API_KEY directly
# TASK_MODEL="azure_ai/gpt-5.5"
# SUPERVISION_MODEL="azure_ai/gpt-5.5"
# BACKEND="claude"

# Preset D: GPT-5.5 proxy (meta) + gpt-oss Tinker (target)
META_MODEL="azure_ai/gpt-5.5"
SUPERVISION_MODEL="azure_ai/gpt-5.5"
TASK_MODEL="openai/gpt-oss-120b"           # base model via Tinker SDK
# # TASK_MODEL="tinker://0526a884-428d-5756-8234-0d66db58a27a:train:0/sampler_weights/000005"  # fine-tuned checkpoint
BACKEND="openhands"

# Preset E: previous default (Gemini meta + gpt-oss target)
# META_MODEL="gemini/gemini-3.1-pro-preview"
# SUPERVISION_MODEL="gemini/gemini-3.1-pro-preview"
# TASK_MODEL="openai/gpt-oss-120b"
# BACKEND="openhands"
# ─────────────────────────────────────────────────────────────────────────────
# Optional: path to a reference solution to seed the gen-0 tree.
# If set and the file exists, it is evaluated and written as the first node before the MCTS loop.
# Example: SEED_SOLUTION="${SCRIPT_DIR}/reference/reference_solution.py"
SEED_SOLUTION=""

# ── CLI args ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_id)                    RUN_ID="$2";                     shift 2 ;;
    --exp_duration_min)          EXP_DURATION_MIN="$2";           shift 2 ;;
    --gen0_evolve_duration_min)  GEN0_EVOLVE_DURATION_MIN="$2";   shift 2 ;;
    --task_model_temperature)    TASK_MODEL_TEMPERATURE="$2";     shift 2 ;;
    --meta_model)                META_MODEL="$2";                 shift 2 ;;
    --task_model)                TASK_MODEL="$2";                 shift 2 ;;
    --backend)                   BACKEND="$2";                    shift 2 ;;
    --supervision_model)         SUPERVISION_MODEL="$2";          shift 2 ;;
    --supervision_interval)      SUPERVISION_INTERVAL="$2";       shift 2 ;;
    --supervision_check_timeout_min) SUPERVISION_CHECK_TIMEOUT_MIN="$2";  shift 2 ;;
    --meta_agent_max_turns)      META_AGENT_MAX_TURNS="$2";        shift 2 ;;
    --seed_solution)             SEED_SOLUTION="$2";              shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
# ─────────────────────────────────────────────────────────────────────────────

export OPENPROBLEMS_CACHE_DIR="${OPENPROBLEMS_CACHE_DIR:-$HOME/.cache/openproblems}"

YELLOW="\033[1;33m"; NC="\033[0m"
printf "${YELLOW}[denoising]${NC} Make sure you have downloaded the data and installed requirements.txt.\n"
printf "${YELLOW}[denoising]${NC} Run './download_data.sh' from this task directory if needed.\n\n"

python orchestration/orchestrator.py \
  --task_dir                   ./tasks/denoising \
  --exp_duration_min           "${EXP_DURATION_MIN}" \
  --gen0_evolve_duration       "$((GEN0_EVOLVE_DURATION_MIN * 60))" \
  --task_model_temperature     "${TASK_MODEL_TEMPERATURE}" \
  --run_id                     "${RUN_ID}" \
  --backend                    "${BACKEND}" \
  --meta_model                 "${META_MODEL}" \
  --task_model                 "${TASK_MODEL}" \
  --supervision_model          "${SUPERVISION_MODEL}" \
  --supervision_interval       "${SUPERVISION_INTERVAL}" \
  --supervision_check_timeout_min "${SUPERVISION_CHECK_TIMEOUT_MIN}" \
  --meta_agent_max_turns       "${META_AGENT_MAX_TURNS}" \
  ${SEED_SOLUTION:+--seed_solution "${SEED_SOLUTION}"}
