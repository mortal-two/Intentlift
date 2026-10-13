#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHONPATH_VALUE="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${INTENTIR_DATASET_ROOT:-}"
PROVIDER_CONFIG="${INTENTIR_PROVIDER_CONFIG:-$REPO_ROOT/llm_providers.local.json}"
EVIDENCE_ROOT="${INTENTIR_EVIDENCE_ROOT:-$REPO_ROOT/../intentir-evidence}"
RUN_ROOT="${INTENTIR_DEMO_RUN_ROOT:-$EVIDENCE_ROOT/rundemo/scrc}"
MAX_LIFT_CALLS="${INTENTIR_MAX_LIFT_CALLS:-3}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Run the real Liger softmax SCRC lifecycle.

The command uses the existing Source -> Lift -> TypeCheck -> BindSource ->
source-bound Replay -> Gate1 -> Gate2 path.  Replay uses responses and the
same Source/Harness-bound artifact.c; it does not generate an independent
target_C.

Usage:
  bash rundemo/03_run_liger_softmax_scrc.sh

Environment overrides:
  INTENTIR_DATASET_ROOT   required existing external Dataset root
  INTENTIR_PROVIDER_CONFIG
                          ignored provider JSON path; never committed
  INTENTIR_PROVIDER_NAME  optional provider name from that JSON
  INTENTIR_EFFORT         optional reasoning effort
  INTENTIR_MAX_LIFT_CALLS logical Lift limit, default 3
  INTENTIR_EVIDENCE_ROOT  relative external evidence root
  INTENTIR_DEMO_RUN_ROOT  explicit demo evidence root override

The provider configuration is required unless OPENAI_BASE_URL and
OPENAI_API_KEY are supplied through the process environment.
EOF
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "unexpected argument: $1" >&2
    exit 2
fi

if [[ -z "$DATASET_ROOT" ]]; then
    echo "Dataset root is required." >&2
    echo "Set INTENTIR_DATASET_ROOT to an existing Dataset root." >&2
    exit 2
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root does not exist: $DATASET_ROOT" >&2
    exit 2
fi

PROVIDER_ARGS=()
if [[ -f "$PROVIDER_CONFIG" ]]; then
    PROVIDER_ARGS+=(--provider-config "$PROVIDER_CONFIG")
elif [[ -n "${OPENAI_BASE_URL:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
    : "Use process-injected provider credentials without creating a config file."
else
    echo "No provider configuration is available." >&2
    echo "Set INTENTIR_PROVIDER_CONFIG to an existing ignored JSON file, or" >&2
    echo "inject OPENAI_BASE_URL and OPENAI_API_KEY for this process." >&2
    exit 2
fi

if [[ -n "${INTENTIR_PROVIDER_NAME:-}" ]]; then
    PROVIDER_ARGS+=(--provider "$INTENTIR_PROVIDER_NAME")
fi

if [[ -n "${INTENTIR_EFFORT:-}" ]]; then
    PROVIDER_ARGS+=(--effort "$INTENTIR_EFFORT")
fi

mkdir -p "$RUN_ROOT"

exec env PYTHONPATH="$PYTHONPATH_VALUE" \
    python "$REPO_ROOT/tools/scrc_frontend.py" \
    --case "high_performance:liger_softmax" \
    --dataset-root "$DATASET_ROOT" \
    "${PROVIDER_ARGS[@]}" \
    --lift-backend miniswe \
    --replay-backend responses \
    --max-lift-calls "$MAX_LIFT_CALLS" \
    --run-root "$RUN_ROOT" \
    --json
