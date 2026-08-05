#!/usr/bin/env bash
# run.sh — run main.py then evaluate_tracking.py for seq_1 and seq_3
#
# Usage:
#   bash run.sh [DATA_ROOT] [OUTPUT_ROOT]
#
# Defaults:
#   DATA_ROOT   = ../public_dev
#   OUTPUT_ROOT = ../outputs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${1:-../public_dev}"
OUTPUT_ROOT="${2:-../outputs}"

run_and_eval() {
    local seq="$1"
    local seq_num="${seq#seq_}"
    local gt="$DATA_ROOT/$seq/gt_answer_seq${seq_num}.csv"

    echo ""
    echo "============================================================"
    echo " Running main.py on $seq"
    echo "============================================================"
    python3 main.py \
        --data-root   "$DATA_ROOT" \
        --seq         "$seq" \
        --output-root "$OUTPUT_ROOT"

    # find the latest result.csv for this seq
    local pred
    pred="$(ls -td "$OUTPUT_ROOT/${seq}_"*/ 2>/dev/null | head -1)result.csv"
    if [ ! -f "$pred" ]; then
        echo "ERROR: Could not find result.csv under $OUTPUT_ROOT/${seq}_*" >&2
        exit 1
    fi

    local eval_out="$OUTPUT_ROOT/${seq}_eval.txt"

    echo ""
    echo "============================================================"
    echo " Evaluating $seq  (pred: $pred)"
    echo "============================================================"
    python3 evaluate_tracking.py \
        --gt   "$gt" \
        --pred "$pred" \
        --out  "$eval_out"

    echo " Evaluation results saved to: $eval_out"
}

run_and_eval seq_1
run_and_eval seq_3

echo ""
echo "============================================================"
echo " Done."
echo "============================================================"
