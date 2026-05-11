#!/usr/bin/env python3
"""Prepare independent EN/KO/FR evaluation JSONL for SIREN language-rank experiments.

This script does NOT translate samples. Each output row is loaded from an actual
language-specific dataset and normalized into:

  {id, source_dataset, source_id, source_lang, lang, text, prompt, response, label, meta}

label: 1 = unsafe / harmful / hateful / toxic / offensive, 0 = safe / benign / neutral.

Supported sources:
  - Hugging Face datasets via `hf_id`
  - local CSV/JSONL via `path`

The normalizer is config-driven so you can add/replace datasets without editing code.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml
from datasets import load_dataset


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    # Fix: os.path.dirname("bare_file.jsonl") == "" -> makedirs("") raises FileNotFoundError
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_value(row: Dict[str, Any], field: Optional[str], default: Any = None) -> Any:
    if field is None:
        return default
    if field in row:
        return row[field]
    cur: Any = row
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def norm_scalar(x: Any) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    return str(x).strip().lower()


def normalize_label(row: Dict[str, Any], spec: Dict[str, Any]) -> Optional[int]:
    """Return 0/1 or None to drop row."""
    if "label_expr" in spec:
        expr = spec["label_expr"]
        if expr == "kmhas_any_hate":
            labels = get_value(row, spec.get("label_field", "label"), [])
            if not isinstance(labels, list):
                labels = [labels]
            safe_values = set(spec.get("safe_values", [8, "8", "Not Hate Speech", "not hate speech"]))
            return 0 if all(x in safe_values or str(x) in safe_values for x in labels) else 1
        if expr == "mlma_normal_vs_rest":
            val = norm_scalar(get_value(row, spec.get("label_field", "tweet sentiment")))
            return 0 if val == "normal" else 1
        if expr == "toxigen_threshold":
            val = get_value(row, spec.get("label_field", "prompt_label"))
            try:
                return 1 if float(val) >= float(spec.get("threshold", 0.5)) else 0
            except Exception:
                return None
        raise ValueError(f"Unknown label_expr: {expr}")

    val = get_value(row, spec.get("label_field"))
    if val is None:
        return None

    if isinstance(val, list):
        unsafe_values = set(map(str, spec.get("unsafe_values", [])))
        safe_values = set(map(str, spec.get("safe_values", [])))
        vals = set(map(str, val))
        if unsafe_values and vals.intersection(unsafe_values):
            return 1
        if safe_values and vals.issubset(safe_values):
            return 0
        return None

    s = norm_scalar(val)
    unsafe_values = set(norm_scalar(x) for x in spec.get("unsafe_values", []))
    safe_values = set(norm_scalar(x) for x in spec.get("safe_values", []))

    if unsafe_values and s in unsafe_values:
        return 1
    if safe_values and s in safe_values:
        return 0

    if "threshold" in spec:
        try:
            return 1 if float(val) >= float(spec["threshold"]) else 0
        except Exception:
            return None

    if s in {"1", "true", "unsafe", "harmful", "toxic", "hateful", "hate", "offensive", "abusive", "haineux"}:
        return 1
    if s in {"0", "false", "safe", "harmless", "benign", "normal", "neutral", "none", "not hate speech", "non_hateful", "non_haineux"}:
        return 0
    return None


def row_passes_filters(row: Dict[str, Any], filters: List[Dict[str, Any]]) -> bool:
    for flt in filters:
        field = flt["field"]
        val = get_value(row, field)
        if "equals" in flt and val != flt["equals"]:
            return False
        if "in" in flt and val not in flt["in"]:
            return False
        if "not_in" in flt and val in flt["not_in"]:
            return False
    return True


# ---------------------------------------------------------------------------
# HF dataset loading — bounded to avoid pulling massive datasets into CPU RAM.
# H100 80 GB note: VRAM is not the constraint here. CPU RAM is.
# civil_comments train has ~1.8M rows; we only need 1 000 samples.
# ---------------------------------------------------------------------------

_LOAD_BUFFER = 10  # load up to N * _LOAD_BUFFER rows to buffer for drop rate


def _hf_load_bounded(hf_id: str, kwargs: Dict[str, Any], split: str, load_limit: int) -> List[Dict[str, Any]]:
    """Load at most load_limit rows. Tries split-slicing first, falls back to streaming."""
    # Strategy 1: HF split slicing — fast, no streaming overhead.
    try:
        dset = load_dataset(
            hf_id, **kwargs,
            split=f"{split}[:{load_limit}]",
            trust_remote_code=True,
        )
        return [dict(x) for x in dset]
    except Exception:
        pass
    # Strategy 2: Streaming — works on all HF datasets.
    dset = load_dataset(hf_id, **kwargs, split=split, streaming=True, trust_remote_code=True)
    rows: List[Dict[str, Any]] = []
    for x in dset:
        rows.append(dict(x))
        if len(rows) >= load_limit:
            break
    return rows


def load_rows(ds_cfg: Dict[str, Any], max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    if ds_cfg.get("hf_id"):
        kwargs: Dict[str, Any] = {}
        if ds_cfg.get("config_name"):
            kwargs["name"] = ds_cfg["config_name"]
        if ds_cfg.get("data_files"):
            kwargs["data_files"] = ds_cfg["data_files"]
        split = ds_cfg.get("split", "train")

        if max_samples is not None:
            return _hf_load_bounded(ds_cfg["hf_id"], kwargs, split, max_samples * _LOAD_BUFFER)

        # No cap — only appropriate for small datasets.
        dset = load_dataset(ds_cfg["hf_id"], **kwargs, split=split, trust_remote_code=True)
        return [dict(x) for x in dset]

    path = ds_cfg.get("path")
    if not path:
        raise ValueError(f"Dataset {ds_cfg.get('name')} needs either hf_id or path")
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep).fillna("").to_dict("records")
    raise ValueError(f"Unsupported local dataset extension: {path}")


def balanced_sample(rows: List[Dict[str, Any]], max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_samples is None or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)

    # Index-based tracking instead of id() — avoids relying on CPython memory
    # identity (id() is unique only for simultaneously live objects in CPython).
    by_label: Dict[int, List[int]] = {0: [], 1: []}
    for i, r in enumerate(rows):
        by_label[int(r["label"])].append(i)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)

    half = max_samples // 2
    selected: List[int] = by_label[0][:half] + by_label[1][: max_samples - half]

    # Top up if one class is short.
    if len(selected) < max_samples:
        used = set(selected)
        rest = [i for i in range(len(rows)) if i not in used]
        rng.shuffle(rest)
        selected.extend(rest[: max_samples - len(selected)])

    rng.shuffle(selected)
    return [rows[i] for i in selected]


def prepare_dataset(ds_cfg: Dict[str, Any], global_seed: int, default_max_samples: Optional[int]) -> List[Dict[str, Any]]:
    name = ds_cfg["name"]
    lang = ds_cfg["lang"]
    max_samples = ds_cfg.get("max_samples", default_max_samples)

    rows = load_rows(ds_cfg, max_samples=max_samples)
    filters = ds_cfg.get("filters") or []
    rows = [r for r in rows if row_passes_filters(r, filters)]

    text_field = ds_cfg.get("text_field", "text")
    id_field = ds_cfg.get("id_field")
    out: List[Dict[str, Any]] = []
    dropped = 0

    for idx, row in enumerate(rows):
        label = normalize_label(row, ds_cfg)
        text = get_value(row, text_field)
        if label is None or text is None or not str(text).strip():
            dropped += 1
            continue
        source_id = str(get_value(row, id_field, idx)) if id_field else str(idx)
        meta = {k: row.get(k) for k in ds_cfg.get("meta_fields", []) if k in row}
        out.append(
            {
                "id": f"{name}::{source_id}",
                "source_dataset": name,
                "source_id": source_id,
                "source_lang": lang,
                "lang": lang,
                "label": int(label),
                "prompt": str(text),
                "response": "",
                "text": str(text),
                "meta": meta,
            }
        )

    out = balanced_sample(out, max_samples, seed=int(ds_cfg.get("seed", global_seed))) if max_samples else out
    n0 = sum(1 for r in out if r["label"] == 0)
    n1 = sum(1 for r in out if r["label"] == 1)
    print(f"{name:28s} lang={lang} kept={len(out)} safe={n0} unsafe={n1} dropped={dropped}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Default: {output_dir}/lang_eval.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exp = cfg["experiment"]
    seed = int(exp.get("seed", 42))
    random.seed(seed)
    default_max_samples = cfg.get("data", {}).get("default_max_samples_per_dataset")
    output_path = args.output or os.path.join(exp["output_dir"], "lang_eval.jsonl")

    all_rows: List[Dict[str, Any]] = []
    for ds_cfg in cfg["datasets"]:
        all_rows.extend(prepare_dataset(ds_cfg, seed, default_max_samples))

    random.Random(seed).shuffle(all_rows)
    write_jsonl(output_path, all_rows)
    print(f"\nSaved {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
