#!/usr/bin/env python3
"""Run SIREN official/local models on trilingual data and extract layer-rank values.

Outputs:
  predictions.jsonl
  layer_values.csv

Core layer statistic:
  For each layer l, selected neurons z_l are multiplied by SIREN's layer weight w_l.
  We save:
    signed_sum_l = sum(z_l * w_l)
    abs_sum_l    = sum(abs(z_l * w_l))   # rank 기준 기본값
    l2_l         = ||z_l * w_l||_2

For local pkl models, this follows CSSLab/SIREN best_model.pkl keys:
  pooling_type, selected_neurons_dict, layer_weights, selected_layers, final_mlp

H100 80 GB note: 8B bfloat16 ~ 16 GB VRAM — well within limit. hf_siren still
loads the backbone twice (SirenGuard then RepresentationExtractor) sequentially,
which is a time cost but not a VRAM issue since the first is deleted before the
second is loaded.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class AdaptiveMLPClassifier(nn.Module):
    """Compatible class for unpickling CSSLab/SIREN final_mlp."""

    def __init__(self, input_dim: int, layer_dims: List[int], dropout_rates: List[float], num_classes: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim, dropout in zip(layer_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "AdaptiveMLPClassifier":
            return AdaptiveMLPClassifier
        return super().find_class(module, name)


def compat_pickle_load(path: str) -> Any:
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return _CompatUnpickler(f).load()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reset_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class RepresentationExtractor:
    """Generic Qwen/Llama hidden-state extractor compatible with CSSLab/SIREN pooling."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        rep_types: Optional[List[str]] = None,
        max_length: int = 512,
    ) -> None:
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.rep_types = rep_types or ["residual_mean", "mlp_mean"]
        self.max_length = max_length
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" and self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": self.device} if self.device != "cpu" else None,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
        self.residual_outputs: List[torch.Tensor] = []
        self.mlp_outputs: List[torch.Tensor] = []
        self.hooks: List[Any] = []

    def _residual_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden_states = output[0].detach() if isinstance(output, tuple) else output.detach()
            while len(self.residual_outputs) <= layer_idx:
                self.residual_outputs.append(torch.empty(0))
            self.residual_outputs[layer_idx] = hidden_states
        return hook

    def _mlp_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            mlp_output = output[0].detach() if isinstance(output, tuple) else output.detach()
            while len(self.mlp_outputs) <= layer_idx:
                self.mlp_outputs.append(torch.empty(0))
            self.mlp_outputs[layer_idx] = mlp_output
        return hook

    def register_hooks(self) -> None:
        for idx, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(self._residual_hook(idx)))
            # Guard: some architectures (MoE, etc.) may not expose a .mlp attribute.
            if hasattr(layer, "mlp"):
                self.hooks.append(layer.mlp.register_forward_hook(self._mlp_hook(idx)))
            else:
                # Ensure mlp_outputs slot exists so indexing stays consistent.
                while len(self.mlp_outputs) <= idx:
                    self.mlp_outputs.append(torch.empty(0))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def close(self) -> None:
        self.remove_hooks()
        del self.model
        cleanup_cuda()

    def extract_batch(self, texts: List[str]) -> List[Dict[int, Dict[str, np.ndarray]]]:
        texts = [t.strip() if str(t).strip() else " " for t in texts]
        self.residual_outputs = []
        self.mlp_outputs = []
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = self.model(**inputs)

        actual_batch_size = inputs["input_ids"].shape[0]
        reps: List[Dict[int, Dict[str, np.ndarray]]] = []
        for b in range(actual_batch_size):
            sample: Dict[int, Dict[str, np.ndarray]] = {}
            valid_len = int(inputs["attention_mask"][b].sum().item())
            for layer_idx in range(self.num_layers):
                layer_rep: Dict[str, np.ndarray] = {}
                if "residual_mean" in self.rep_types:
                    residual = self.residual_outputs[layer_idx]
                    if residual.dim() == 2:
                        residual = residual.unsqueeze(0)
                    if residual.shape[0] != actual_batch_size and residual.shape[1] == actual_batch_size:
                        residual = residual.transpose(0, 1)
                    layer_rep["residual_mean"] = residual[b, :valid_len].mean(dim=0).cpu().float().numpy()
                if "mlp_mean" in self.rep_types and layer_idx < len(self.mlp_outputs):
                    mlp = self.mlp_outputs[layer_idx]
                    if mlp.numel() > 0:
                        if mlp.dim() == 2:
                            mlp = mlp.unsqueeze(0)
                        if mlp.shape[0] != actual_batch_size and mlp.shape[1] == actual_batch_size:
                            mlp = mlp.transpose(0, 1)
                        layer_rep["mlp_mean"] = mlp[b, :valid_len].mean(dim=0).cpu().float().numpy()
                sample[layer_idx] = layer_rep
            reps.append(sample)
        return reps


@dataclass
class SirenMeta:
    pooling_type: str
    selected_neurons_dict: Dict[str, List[int]]
    layer_weights: Dict[int, float]
    selected_layers: List[int]
    final_mlp: Optional[nn.Module] = None


def _first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return default


def _normalize_layer_weights(layer_weights: Any) -> Dict[int, float]:
    if layer_weights is None:
        return {}
    if isinstance(layer_weights, list):
        return {i: float(v) for i, v in enumerate(layer_weights)}
    return {int(str(k).replace("layer", "")): float(v) for k, v in dict(layer_weights).items()}


def _infer_pooling_type(selected_neurons: Dict[str, Any], fallback: str = "residual_mean") -> str:
    for k in selected_neurons.keys():
        if "residual_mean" in k:
            return "residual_mean"
        if "mlp_mean" in k:
            return "mlp_mean"
    return fallback


def _normalize_selected_neurons(raw: Any, pooling_type: str) -> Dict[str, List[int]]:
    if raw is None:
        return {}
    out: Dict[str, List[int]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if str(k).startswith("layer") and ("residual_mean" in str(k) or "mlp_mean" in str(k)):
                out[str(k)] = [int(x) for x in list(v)]
            elif str(k).lstrip("-").isdigit():
                out[f"layer{int(k)}_{pooling_type}"] = [int(x) for x in list(v)]
            elif isinstance(v, dict):
                nested_pooling = str(k)
                for lk, lv in v.items():
                    if str(lk).lstrip("-").isdigit():
                        out[f"layer{int(lk)}_{nested_pooling}"] = [int(x) for x in list(lv)]
    return out


def load_pkl_siren_meta(pkl_path: str, device: str) -> SirenMeta:
    obj = compat_pickle_load(pkl_path)
    if isinstance(obj, dict) and "best_overall" in obj:
        obj = obj["best_overall"]
    pooling_type = str(obj.get("pooling_type", "residual_mean"))
    selected_neurons_dict = _normalize_selected_neurons(obj.get("selected_neurons_dict"), pooling_type)
    layer_weights = _normalize_layer_weights(obj.get("layer_weights"))
    selected_layers = [int(x) for x in obj.get("selected_layers", sorted(layer_weights.keys()))]
    final_mlp = obj.get("final_mlp")
    if final_mlp is not None:
        final_mlp = final_mlp.to(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        final_mlp.eval()
    return SirenMeta(pooling_type, selected_neurons_dict, layer_weights, selected_layers, final_mlp)


def load_hf_siren_meta(repo_id: str, fallback_base_model: Optional[str] = None) -> Tuple[SirenMeta, str]:
    config_path = hf_hub_download(repo_id=repo_id, filename="siren_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    candidates = [cfg]
    for key in ["siren", "classifier", "model_config"]:
        if isinstance(cfg.get(key), dict):
            candidates.append(cfg[key])

    merged: Dict[str, Any] = {}
    for c in candidates:
        merged.update(c)

    raw_selected = _first_present(
        merged,
        [
            "selected_neurons_dict",
            "selected_neurons",
            "safety_neuron_indices",
            "per_layer_safety_neuron_indices",
            "neuron_indices",
        ],
    )
    pooling_type = str(_first_present(merged, ["pooling_type", "pooling", "rep_type"], ""))
    if not pooling_type:
        pooling_type = _infer_pooling_type(raw_selected or {}, "residual_mean")
    selected_neurons_dict = _normalize_selected_neurons(raw_selected, pooling_type)
    layer_weights = _normalize_layer_weights(_first_present(merged, ["layer_weights", "performance_layer_weights", "weights_by_layer"]))
    selected_layers_raw = _first_present(merged, ["selected_layers", "layers"], None)
    if selected_layers_raw is None:
        selected_layers = sorted(layer_weights.keys())
    else:
        selected_layers = [int(x) for x in selected_layers_raw]

    base_model = _first_present(
        merged,
        [
            "base_model",
            "base_model_id",
            "base_model_name",
            "base_model_name_or_path",
            "backbone",
            "backbone_model",
            "model_path",
        ],
        fallback_base_model,
    )
    if base_model is None:
        raise KeyError(
            f"Cannot find base model in {repo_id}/siren_config.json. "
            f"Add base_model manually in configs/exp_lang_rank.yaml. Config keys={list(merged.keys())}"
        )
    if not selected_neurons_dict or not layer_weights:
        raise KeyError(
            f"Cannot parse selected_neurons/layer_weights from {repo_id}/siren_config.json. "
            f"Config keys={list(merged.keys())}"
        )
    return SirenMeta(pooling_type, selected_neurons_dict, layer_weights, selected_layers, None), str(base_model)


def get_text(row: Dict[str, Any], mode: str) -> str:
    if mode == "text":
        return str(row.get("text", ""))
    prompt = str(row.get("prompt", ""))
    response = str(row.get("response", ""))
    if response:
        return f"{prompt}\n{response}".strip()
    return str(row.get("text", prompt)).strip()


def selected_key(layer_idx: int, pooling_type: str, selected_neurons_dict: Dict[str, List[int]]) -> Optional[str]:
    # Fix: prefer exact pooling_type match before falling back to any prefix match.
    # Previously, dict iteration order could silently pick the wrong pooling type
    # (e.g. mlp_mean instead of residual_mean) when both exist for the same layer.
    exact = f"layer{layer_idx}_{pooling_type}"
    if exact in selected_neurons_dict:
        return exact
    # Fallback: any key for this layer index.
    for k in selected_neurons_dict:
        if k.startswith(f"layer{layer_idx}_"):
            return k
    return None


def layer_values_for_reps(
    reps: List[Dict[int, Dict[str, np.ndarray]]],
    meta: SirenMeta,
    row_metas: List[Dict[str, Any]],
    model_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_rep, row in zip(reps, row_metas):
        for layer_idx in meta.selected_layers:
            key = selected_key(layer_idx, meta.pooling_type, meta.selected_neurons_dict)
            if key is None:
                continue
            pooling = key.split("_", 1)[1]
            if pooling not in sample_rep[layer_idx]:
                pooling = meta.pooling_type
            # Guard: if pooling type is still missing (e.g. model has no .mlp),
            # skip this layer rather than raising KeyError.
            if pooling not in sample_rep[layer_idx]:
                continue
            selected_indices = meta.selected_neurons_dict[key]
            layer_features = sample_rep[layer_idx][pooling]
            selected = layer_features[selected_indices]
            weight = float(meta.layer_weights.get(layer_idx, meta.layer_weights.get(str(layer_idx), 1.0)))
            weighted = selected * weight
            rows.append(
                {
                    "model": model_name,
                    "id": row["id"],
                    "source_dataset": row["source_dataset"],
                    "source_id": row["source_id"],
                    "source_lang": row["source_lang"],
                    "lang": row["lang"],
                    "label": int(row["label"]),
                    "layer_idx": int(layer_idx),
                    "pooling_type": pooling,
                    "layer_weight": weight,
                    "n_selected_neurons": int(len(selected_indices)),
                    "raw_signed_sum": float(np.sum(selected)),
                    "raw_abs_sum": float(np.sum(np.abs(selected))),
                    "raw_l2": float(np.linalg.norm(selected)),
                    "raw_mean_abs": float(np.mean(np.abs(selected))) if len(selected) else 0.0,
                    "signed_sum": float(np.sum(weighted)),
                    "abs_sum": float(np.sum(np.abs(weighted))),
                    "l2": float(np.linalg.norm(weighted)),
                    "mean_abs": float(np.mean(np.abs(weighted))) if len(weighted) else 0.0,
                }
            )
    return rows


def aggregate_features(reps: List[Dict[int, Dict[str, np.ndarray]]], meta: SirenMeta) -> np.ndarray:
    all_features: List[np.ndarray] = []
    for sample_rep in reps:
        chunks: List[np.ndarray] = []
        for layer_idx in meta.selected_layers:
            key = selected_key(layer_idx, meta.pooling_type, meta.selected_neurons_dict)
            if key is None:
                continue
            pooling = key.split("_", 1)[1]
            if pooling not in sample_rep[layer_idx]:
                pooling = meta.pooling_type
            if pooling not in sample_rep[layer_idx]:
                continue
            indices = meta.selected_neurons_dict[key]
            weight = float(meta.layer_weights.get(layer_idx, meta.layer_weights.get(str(layer_idx), 1.0)))
            chunks.append(sample_rep[layer_idx][pooling][indices] * weight)
        all_features.append(np.concatenate(chunks))
    return np.asarray(all_features, dtype=np.float32)


def predict_with_local_mlp(X: np.ndarray, meta: SirenMeta, device: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if meta.final_mlp is None:
        raise ValueError("pkl_siren requires final_mlp in best_model.pkl")
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    preds: List[int] = []
    scores: List[float] = []
    meta.final_mlp.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.from_numpy(X[i : i + batch_size]).to(device)
            logits = meta.final_mlp(bx)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.detach().cpu().float().numpy().tolist())
            preds.extend((probs >= 0.5).long().detach().cpu().numpy().tolist())
    return np.asarray(scores), np.asarray(preds)


def score_with_hf_runtime(repo_id: str, texts: List[str], device: str, dtype: str, threshold: float, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from siren_guard import SirenGuard
    except ImportError as e:
        raise ImportError("Install official runtime first: pip install llm-siren") from e
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else torch.float32
    guard = SirenGuard.from_pretrained(repo_id, device=device, dtype=torch_dtype)
    scores: List[float] = []
    preds: List[int] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"score {repo_id}"):
        batch = texts[i : i + batch_size]
        results = guard.score_batch(batch, threshold=threshold)
        for r in results:
            scores.append(float(r.score))
            preds.append(int(r.is_harmful))
    del guard
    cleanup_cuda()
    return np.asarray(scores), np.asarray(preds)


def run_model(model_cfg: Dict[str, Any], data: List[Dict[str, Any]], cfg: Dict[str, Any], pred_path: str, layer_path: str, skip_layer_values: bool) -> None:
    exp = cfg["experiment"]
    model_name = model_cfg["name"]
    kind = model_cfg["kind"]
    device = exp.get("device", "cuda")
    dtype = exp.get("dtype", "bfloat16")
    batch_size = int(exp.get("batch_size", 16))
    threshold = float(exp.get("threshold", 0.5))
    max_length = int(exp.get("max_length", 512))
    mode = exp.get("mode", "prompt_response")
    texts = [get_text(r, mode) for r in data]

    print(f"\n=== Running {model_name} ({kind}) ===")

    if kind == "hf_siren":
        repo_id = model_cfg["repo_id"]
        scores, preds = score_with_hf_runtime(repo_id, texts, device, dtype, threshold, batch_size)
        meta, base_model = load_hf_siren_meta(repo_id, model_cfg.get("base_model"))
    elif kind == "pkl_siren":
        meta = load_pkl_siren_meta(model_cfg["pkl_path"], device)
        base_model = model_cfg["base_model"]
        scores = np.full(len(data), np.nan, dtype=float)
        preds = np.full(len(data), -1, dtype=int)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    if skip_layer_values and kind == "hf_siren":
        pred_rows = []
        for row, score, pred in zip(data, scores, preds):
            pred_rows.append(
                {
                    "model": model_name,
                    "id": row["id"],
                    "source_dataset": row["source_dataset"],
                    "source_id": row["source_id"],
                    "source_lang": row["source_lang"],
                    "lang": row["lang"],
                    "label": int(row["label"]),
                    "score": float(score),
                    "pred": int(pred),
                }
            )
        append_jsonl(pred_path, pred_rows)
        return

    rep_types = list({meta.pooling_type, "residual_mean"})
    extractor = RepresentationExtractor(base_model, device=device, dtype=dtype, rep_types=rep_types, max_length=max_length)
    extractor.register_hooks()

    all_pred_rows: List[Dict[str, Any]] = []
    layer_rows_accum: List[Dict[str, Any]] = []
    header_written = os.path.exists(layer_path) and os.path.getsize(layer_path) > 0

    for start in tqdm(range(0, len(data), batch_size), desc=f"extract {model_name}"):
        end = start + batch_size
        batch_rows = data[start:end]
        batch_texts = texts[start:end]

        try:
            reps = extractor.extract_batch(batch_texts)
        except torch.cuda.OutOfMemoryError:
            print(
                f"\n[OOM] Batch {start}-{end} failed. "
                f"Reduce batch_size in config (currently {batch_size}). "
                f"H100 80 GB is ample for 8B bfloat16 at batch_size<=32; "
                f"check for other processes holding VRAM.",
                file=sys.stderr,
            )
            raise

        if kind == "pkl_siren":
            X = aggregate_features(reps, meta)
            b_scores, b_preds = predict_with_local_mlp(X, meta, device, batch_size=batch_size)
        else:
            b_scores = scores[start:end]
            b_preds = preds[start:end]

        for row, score, pred in zip(batch_rows, b_scores, b_preds):
            all_pred_rows.append(
                {
                    "model": model_name,
                    "id": row["id"],
                    "source_dataset": row["source_dataset"],
                    "source_id": row["source_id"],
                    "source_lang": row["source_lang"],
                    "lang": row["lang"],
                    "label": int(row["label"]),
                    "score": float(score),
                    "pred": int(pred),
                }
            )

        if not skip_layer_values:
            layer_rows = layer_values_for_reps(reps, meta, batch_rows, model_name)
            layer_rows_accum.extend(layer_rows)
            if len(layer_rows_accum) >= 5000:
                pd.DataFrame(layer_rows_accum).to_csv(layer_path, mode="a", index=False, header=not header_written)
                header_written = True
                layer_rows_accum = []

    extractor.close()
    append_jsonl(pred_path, all_pred_rows)
    if layer_rows_accum:
        pd.DataFrame(layer_rows_accum).to_csv(layer_path, mode="a", index=False, header=not header_written)
    cleanup_cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None, help="Default: {output_dir}/lang_eval.jsonl")
    parser.add_argument("--skip_layer_values", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["experiment"]["output_dir"]
    data_path = args.data or os.path.join(output_dir, "lang_eval.jsonl")
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    layer_path = os.path.join(output_dir, "layer_values.csv")
    data = read_jsonl(data_path)

    reset_file(pred_path)
    if not args.skip_layer_values:
        reset_file(layer_path)

    for model_cfg in cfg["models"]:
        try:
            run_model(model_cfg, data, cfg, pred_path, layer_path, args.skip_layer_values)
        except Exception as e:
            print(f"[ERROR] {model_cfg.get('name')} failed: {type(e).__name__}: {e}", file=sys.stderr)
            raise

    print(f"\nSaved predictions: {pred_path}")
    if not args.skip_layer_values:
        print(f"Saved layer values: {layer_path}")


if __name__ == "__main__":
    main()
