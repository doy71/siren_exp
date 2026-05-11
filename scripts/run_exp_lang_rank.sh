#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/exp_lang_rank.yaml}
OUT_DIR=$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$CONFIG", encoding="utf-8"))
print(cfg["experiment"]["output_dir"])
PY
)
mkdir -p "$OUT_DIR"
python scripts/prepare_independent_lang_datasets.py --config "$CONFIG" --output "$OUT_DIR/lang_eval.jsonl"
python scripts/run_siren_exp.py --config "$CONFIG" --data "$OUT_DIR/lang_eval.jsonl"
python scripts/analyze_lang_rank.py --config "$CONFIG"
