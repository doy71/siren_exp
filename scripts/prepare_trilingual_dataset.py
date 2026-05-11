#!/usr/bin/env python3
"""Prepare EN/KO/FR evaluation JSONL for SIREN language-rank experiments.

Input records can be either:
  {"id": "...", "text": "...", "label": 0/1}
  {"id": "...", "prompt": "...", "response": "...", "label": 0/1}

Output records contain source_id, source_dataset, source_lang, lang, prompt/response/text, label.
For English-source rows, it creates en/ko/fr. For Korean-source rows, it creates ko/en/fr.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


TARGET_LANGS = ["en", "ko", "fr"]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "unsafe", "harmful", "bad", "yes", "true"}:
        return 1
    if s in {"0", "safe", "harmless", "good", "no", "false"}:
        return 0
    raise ValueError(f"Cannot normalize label: {x!r}")


def record_text(row: Dict[str, Any]) -> str:
    if row.get("text") is not None:
        return str(row["text"])
    prompt = str(row.get("prompt", ""))
    response = str(row.get("response", ""))
    if response:
        return f"{prompt}\n{response}".strip()
    return prompt.strip()


class NLLBTranslator:
    def __init__(self, model_name: str, lang_codes: Dict[str, str], device: str, dtype: str = "bfloat16") -> None:
        self.model_name = model_name
        self.lang_codes = lang_codes
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" and self.device != "cpu" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(self.device)
        self.model.eval()

    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str, max_new_tokens: int = 512) -> List[str]:
        if src_lang == tgt_lang:
            return texts
        src_code = self.lang_codes[src_lang]
        tgt_code = self.lang_codes[tgt_lang]
        self.tokenizer.src_lang = src_code
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_field(
    translator: Optional[NLLBTranslator],
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    if src_lang == tgt_lang:
        return texts
    if translator is None:
        raise RuntimeError("translation.enabled=false인데 target language row가 필요합니다.")
    outs: List[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"translate {src_lang}->{tgt_lang}"):
        batch = texts[i : i + batch_size]
        outs.extend(translator.translate_batch(batch, src_lang, tgt_lang, max_new_tokens=max_new_tokens))
    return outs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Default: {output_dir}/trilingual_eval.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    random.seed(seed)

    exp_cfg = cfg["experiment"]
    trans_cfg = cfg.get("translation", {})
    output_path = args.output or os.path.join(exp_cfg["output_dir"], "trilingual_eval.jsonl")

    translator: Optional[NLLBTranslator] = None
    if trans_cfg.get("enabled", True):
        translator = NLLBTranslator(
            model_name=trans_cfg.get("model_name", "facebook/nllb-200-distilled-600M"),
            lang_codes=trans_cfg.get("lang_codes", {"en": "eng_Latn", "ko": "kor_Hang", "fr": "fra_Latn"}),
            device=exp_cfg.get("device", "cuda"),
            dtype=exp_cfg.get("dtype", "bfloat16"),
        )

    out_rows: List[Dict[str, Any]] = []
    for ds in cfg["datasets"]:
        rows = read_jsonl(ds["path"])
        if ds.get("max_samples") is not None:
            rows = rows[: int(ds["max_samples"])]
        source_lang = ds["source_lang"]
        if source_lang not in {"en", "ko"}:
            raise ValueError("This experiment expects source_lang to be either 'en' or 'ko'.")

        base_records: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            rid = str(row.get("id", f"{ds['name']}_{idx}"))
            label = normalize_label(row.get("label", row.get("is_unsafe", row.get("unsafe", row.get("harmful")))))
            base_records.append(
                {
                    "source_dataset": ds["name"],
                    "source_id": rid,
                    "source_lang": source_lang,
                    "label": label,
                    "prompt": str(row.get("prompt", "")),
                    "response": str(row.get("response", "")),
                    "text": record_text(row),
                    "meta": row.get("meta", {}),
                }
            )

        prompts = [r["prompt"] for r in base_records]
        responses = [r["response"] for r in base_records]
        texts = [r["text"] for r in base_records]

        for tgt_lang in TARGET_LANGS:
            if tgt_lang == source_lang:
                tgt_prompts, tgt_responses, tgt_texts = prompts, responses, texts
            else:
                if any(p.strip() for p in prompts) or any(r.strip() for r in responses):
                    tgt_prompts = translate_field(
                        translator,
                        prompts,
                        source_lang,
                        tgt_lang,
                        int(trans_cfg.get("batch_size", 16)),
                        int(trans_cfg.get("max_new_tokens", 512)),
                    )
                    tgt_responses = translate_field(
                        translator,
                        responses,
                        source_lang,
                        tgt_lang,
                        int(trans_cfg.get("batch_size", 16)),
                        int(trans_cfg.get("max_new_tokens", 512)),
                    )
                    tgt_texts = [f"{p}\n{r}".strip() if r else p.strip() for p, r in zip(tgt_prompts, tgt_responses)]
                else:
                    tgt_texts = translate_field(
                        translator,
                        texts,
                        source_lang,
                        tgt_lang,
                        int(trans_cfg.get("batch_size", 16)),
                        int(trans_cfg.get("max_new_tokens", 512)),
                    )
                    tgt_prompts = tgt_texts
                    tgt_responses = [""] * len(tgt_texts)

            for base, p, r, t in zip(base_records, tgt_prompts, tgt_responses, tgt_texts):
                out_rows.append(
                    {
                        "id": f"{base['source_dataset']}::{base['source_id']}::{tgt_lang}",
                        "source_dataset": base["source_dataset"],
                        "source_id": base["source_id"],
                        "source_lang": base["source_lang"],
                        "lang": tgt_lang,
                        "label": base["label"],
                        "prompt": p,
                        "response": r,
                        "text": t,
                        "meta": base["meta"],
                    }
                )

    write_jsonl(output_path, out_rows)
    print(f"Saved {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
