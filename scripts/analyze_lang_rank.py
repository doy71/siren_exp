#!/usr/bin/env python3
"""Analyze SIREN EN/KO/FR predictions, layer-rank shifts, and layer-scale shifts.

Important interpretation:
  - SIREN's layer_weights are fixed learned/performance weights for a given model.
  - Language-dependent changes come from selected-neuron activations.
  - weighted_* metrics are selected activations multiplied by fixed layer_weight.
  - These metrics are diagnostic contribution proxies before the final MLP, not exact
    causal attributions of the final score.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def read_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def safe_auprc(y_true, y_score):
    if len(set(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def compute_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "source_dataset", "source_lang", "lang"]
    for keys, g in pred.groupby(group_cols):
        y = g["label"].astype(int).to_numpy()
        p = g["pred"].astype(int).to_numpy()
        s = g["score"].astype(float).to_numpy()
        rows.append(
            dict(
                zip(group_cols, keys),
                n=len(g),
                accuracy=accuracy_score(y, p),
                precision_unsafe=precision_score(y, p, pos_label=1, zero_division=0),
                recall_unsafe=recall_score(y, p, pos_label=1, zero_division=0),
                f1_unsafe=f1_score(y, p, pos_label=1, zero_division=0),
                f1_macro=f1_score(y, p, average="macro", zero_division=0),
                auroc=safe_auc(y, s) if not np.isnan(s).all() else np.nan,
                auprc=safe_auprc(y, s) if not np.isnan(s).all() else np.nan,
            )
        )
    return pd.DataFrame(rows).sort_values(group_cols)


def compute_consistency(pred: pd.DataFrame) -> pd.DataFrame:
    """Distributional language comparison for independent datasets.

    Since EN/KO/FR samples are not translations of the same source examples, exact
    sample-level flip-rate is invalid. This reports aggregate shifts instead.
    """
    rows = []
    group = (
        pred.groupby(["model", "lang"])
        .agg(
            n=("pred", "size"),
            unsafe_label_rate=("label", "mean"),
            unsafe_pred_rate=("pred", "mean"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    for model, g in group.groupby("model"):
        by_lang = {r["lang"]: r for _, r in g.iterrows()}
        for a, b in [("en", "ko"), ("en", "fr"), ("ko", "fr")]:
            if a not in by_lang or b not in by_lang:
                continue
            ra, rb = by_lang[a], by_lang[b]
            rows.append(
                {
                    "model": model,
                    "pair": f"{a}-{b}",
                    "n_a": int(ra["n"]),
                    "n_b": int(rb["n"]),
                    "unsafe_label_rate_a": float(ra["unsafe_label_rate"]),
                    "unsafe_label_rate_b": float(rb["unsafe_label_rate"]),
                    "unsafe_pred_rate_a": float(ra["unsafe_pred_rate"]),
                    "unsafe_pred_rate_b": float(rb["unsafe_pred_rate"]),
                    "pred_rate_delta_b_minus_a": float(rb["unsafe_pred_rate"] - ra["unsafe_pred_rate"]),
                    "mean_score_a": float(ra["mean_score"]),
                    "mean_score_b": float(rb["mean_score"]),
                    "mean_score_delta_b_minus_a": float(rb["mean_score"] - ra["mean_score"]),
                }
            )
    return pd.DataFrame(rows)


def _metric_column(rank_metric: str) -> str:
    return {
        "abs_sum": "mean_weighted_abs_sum",
        "signed_sum": "mean_weighted_signed_sum",
        "l2": "mean_weighted_l2",
        "mean_abs": "mean_weighted_mean_abs",
    }[rank_metric]


def _ensure_raw_columns(layer: pd.DataFrame) -> pd.DataFrame:
    """Backwards compatibility with older layer_values.csv files."""
    out = layer.copy()
    aliases = {
        "raw_signed_sum": "signed_sum",
        "raw_abs_sum": "abs_sum",
        "raw_l2": "l2",
        "raw_mean_abs": "mean_abs",
    }
    for raw_col, weighted_col in aliases.items():
        if raw_col not in out.columns:
            out[raw_col] = np.nan
    return out


def compute_layer_tables(layer: pd.DataFrame, rank_metric: str = "abs_sum", eps: float = 1e-12) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer = _ensure_raw_columns(layer)

    # 1) Dataset-level mean scale per layer.
    group_cols = ["model", "source_dataset", "lang", "layer_idx"]
    summary = (
        layer.groupby(group_cols)
        .agg(
            mean_raw_abs_sum=("raw_abs_sum", "mean"),
            mean_raw_signed_sum=("raw_signed_sum", "mean"),
            mean_raw_l2=("raw_l2", "mean"),
            mean_raw_mean_abs=("raw_mean_abs", "mean"),
            mean_weighted_abs_sum=("abs_sum", "mean"),
            mean_weighted_signed_sum=("signed_sum", "mean"),
            mean_weighted_l2=("l2", "mean"),
            mean_weighted_mean_abs=("mean_abs", "mean"),
            layer_weight=("layer_weight", "first"),
            n_selected_neurons=("n_selected_neurons", "first"),
            n_examples=("id", "nunique"),
        )
        .reset_index()
    )

    metric_col = _metric_column(rank_metric)
    summary["rank"] = summary.groupby(["model", "source_dataset", "lang"])[metric_col].rank(method="min", ascending=False)
    summary["weighted_abs_share"] = summary["mean_weighted_abs_sum"] / summary.groupby(["model", "source_dataset", "lang"])["mean_weighted_abs_sum"].transform(lambda x: x.sum() + eps)
    summary["raw_abs_share"] = summary["mean_raw_abs_sum"] / summary.groupby(["model", "source_dataset", "lang"])["mean_raw_abs_sum"].transform(lambda x: x.sum() + eps)

    # 2) Language-level profile: average across datasets in the same language.
    lang_profile = (
        summary.groupby(["model", "lang", "layer_idx"])
        .agg(
            mean_raw_abs_sum=("mean_raw_abs_sum", "mean"),
            mean_raw_signed_sum=("mean_raw_signed_sum", "mean"),
            mean_raw_l2=("mean_raw_l2", "mean"),
            mean_raw_mean_abs=("mean_raw_mean_abs", "mean"),
            mean_weighted_abs_sum=("mean_weighted_abs_sum", "mean"),
            mean_weighted_signed_sum=("mean_weighted_signed_sum", "mean"),
            mean_weighted_l2=("mean_weighted_l2", "mean"),
            mean_weighted_mean_abs=("mean_weighted_mean_abs", "mean"),
            layer_weight=("layer_weight", "first"),
            n_selected_neurons=("n_selected_neurons", "first"),
            n_examples=("n_examples", "sum"),
        )
        .reset_index()
    )
    lang_profile["rank"] = lang_profile.groupby(["model", "lang"])[metric_col].rank(method="min", ascending=False)
    lang_profile["weighted_abs_share"] = lang_profile["mean_weighted_abs_sum"] / lang_profile.groupby(["model", "lang"])["mean_weighted_abs_sum"].transform(lambda x: x.sum() + eps)
    lang_profile["raw_abs_share"] = lang_profile["mean_raw_abs_sum"] / lang_profile.groupby(["model", "lang"])["mean_raw_abs_sum"].transform(lambda x: x.sum() + eps)
    lang_profile["source_dataset"] = "__LANG_AVG__"

    summary_all = pd.concat([summary, lang_profile[summary.columns]], ignore_index=True)

    # 3) Rank/value shift between language-level average profiles.
    # Fix: skip self-comparisons (lang == base_lang) — they produce trivially
    # zero rank_delta / log_ratio rows that add noise to the output CSVs.
    rank_shifts = []
    scale_shifts = []
    for model, g in lang_profile.groupby("model"):
        by_lang = {lang: lg.copy() for lang, lg in g.groupby("lang")}
        for base_lang in sorted(by_lang):
            # Build baseline columns explicitly instead of using rename().
            #
            # Why: when rank_metric == "abs_sum", metric_col is
            # "mean_weighted_abs_sum". If we use a rename dict like
            # {metric_col: "base_value", "mean_weighted_abs_sum":
            # "base_weighted_abs_sum"}, the duplicate key collapses and
            # "base_value" is never created, causing KeyError: 'base_value'.
            base_src_cols = [
                "layer_idx",
                "rank",
                metric_col,
                "mean_weighted_abs_sum",
                "mean_raw_abs_sum",
                "weighted_abs_share",
                "raw_abs_share",
            ]
            # Remove duplicates while preserving order.
            base_src_cols = list(dict.fromkeys(base_src_cols))
            base_src = by_lang[base_lang][base_src_cols].copy()
            base = pd.DataFrame({
                "layer_idx": base_src["layer_idx"],
                "base_rank": base_src["rank"],
                "base_value": base_src[metric_col],
                "base_weighted_abs_sum": base_src["mean_weighted_abs_sum"],
                "base_raw_abs_sum": base_src["mean_raw_abs_sum"],
                "base_weighted_abs_share": base_src["weighted_abs_share"],
                "base_raw_abs_share": base_src["raw_abs_share"],
            })
            for lang, lg in by_lang.items():
                # Skip self-comparison: rank_delta=0, log_ratio=0 rows are noise.
                if lang == base_lang:
                    continue
                merged = lg.merge(base, on="layer_idx", how="inner")
                merged["model"] = model
                merged["baseline_lang"] = base_lang
                merged["compare_lang"] = lang
                merged["rank_delta"] = merged["rank"] - merged["base_rank"]
                merged["value_delta"] = merged[metric_col] - merged["base_value"]
                rank_shifts.append(merged)

                scale = merged[[
                    "model", "baseline_lang", "compare_lang", "layer_idx", "layer_weight", "n_selected_neurons",
                    "base_rank", "rank", "rank_delta",
                    "base_weighted_abs_sum", "mean_weighted_abs_sum",
                    "base_raw_abs_sum", "mean_raw_abs_sum",
                    "base_weighted_abs_share", "weighted_abs_share",
                    "base_raw_abs_share", "raw_abs_share",
                ]].copy()
                scale = scale.rename(columns={
                    "rank": "compare_rank",
                    "mean_weighted_abs_sum": "compare_weighted_abs_sum",
                    "mean_raw_abs_sum": "compare_raw_abs_sum",
                    "weighted_abs_share": "compare_weighted_abs_share",
                    "raw_abs_share": "compare_raw_abs_share",
                })
                scale["weighted_abs_log_ratio"] = np.log((scale["compare_weighted_abs_sum"] + eps) / (scale["base_weighted_abs_sum"] + eps))
                scale["raw_abs_log_ratio"] = np.log((scale["compare_raw_abs_sum"] + eps) / (scale["base_raw_abs_sum"] + eps))
                scale["weighted_share_delta"] = scale["compare_weighted_abs_share"] - scale["base_weighted_abs_share"]
                scale["raw_share_delta"] = scale["compare_raw_abs_share"] - scale["base_raw_abs_share"]
                scale_shifts.append(scale)

    rank_shift_df = pd.concat(rank_shifts, ignore_index=True) if rank_shifts else pd.DataFrame()
    scale_shift_df = pd.concat(scale_shifts, ignore_index=True) if scale_shifts else pd.DataFrame()

    # 4) Spearman rank correlation between language-level rank profiles.
    corr_rows = []
    for model, g in lang_profile.groupby("model"):
        by_lang = {lang: lg.sort_values("layer_idx") for lang, lg in g.groupby("lang")}
        for a, b in [("en", "ko"), ("en", "fr"), ("ko", "fr")]:
            if a not in by_lang or b not in by_lang:
                continue
            merged = by_lang[a][["layer_idx", "rank", metric_col]].merge(
                by_lang[b][["layer_idx", "rank", metric_col]],
                on="layer_idx",
                suffixes=(f"_{a}", f"_{b}"),
            )
            if len(merged) < 2:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(merged[f"rank_{a}"], merged[f"rank_{b}"])
            corr_rows.append(
                {
                    "model": model,
                    "pair": f"{a}-{b}",
                    "spearman_rank_corr": rho,
                    "p_value": pval,
                    "n_layers": len(merged),
                    "mean_abs_value_delta": float(np.mean(np.abs(merged[f"{metric_col}_{b}"] - merged[f"{metric_col}_{a}"]))) if len(merged) else np.nan,
                }
            )
    corr_df = pd.DataFrame(corr_rows)

    scale_summary = summary_all.rename(columns={
        "mean_weighted_abs_sum": "weighted_abs_mean",
        "mean_weighted_l2": "weighted_l2_mean",
        "mean_raw_abs_sum": "raw_abs_mean",
        "mean_raw_l2": "raw_l2_mean",
        "weighted_abs_share": "normalized_weighted_share",
        "raw_abs_share": "normalized_raw_share",
    })
    return summary_all, rank_shift_df, corr_df, scale_summary, scale_shift_df


def plot_layer_values(summary: pd.DataFrame, output_dir: str, metric_col: str = "mean_weighted_abs_sum") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for (model, ds), g in summary.groupby(["model", "source_dataset"]):
        plt.figure(figsize=(10, 5))
        for lang, lg in g.groupby("lang"):
            lg = lg.sort_values("layer_idx")
            plt.plot(lg["layer_idx"], lg[metric_col], marker="o", label=lang)
        plt.title(f"{model} / {ds} / {metric_col}")
        plt.xlabel("Layer index")
        plt.ylabel(metric_col)
        plt.legend()
        plt.tight_layout()
        safe_model = str(model).replace("/", "_").replace(" ", "_")
        safe_ds = str(ds).replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"layer_value_{safe_model}_{safe_ds}.png"), dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank_metric", default="abs_sum", choices=["abs_sum", "signed_sum", "l2", "mean_abs"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = cfg["experiment"]["output_dir"]
    pred_path = os.path.join(out, "predictions.jsonl")
    layer_path = os.path.join(out, "layer_values.csv")
    analysis_dir = os.path.join(out, "analysis")
    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(analysis_dir, exist_ok=True)

    pred = read_jsonl(pred_path)
    metrics = compute_metrics(pred)
    consistency = compute_consistency(pred)
    metrics.to_csv(os.path.join(analysis_dir, "metrics.csv"), index=False)
    consistency.to_csv(os.path.join(analysis_dir, "consistency.csv"), index=False)

    print("\n=== Metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== Distributional language comparison ===")
    print(consistency.to_string(index=False) if not consistency.empty else "No consistency rows")

    if os.path.exists(layer_path) and os.path.getsize(layer_path) > 0:
        layer = pd.read_csv(layer_path)
        summary, rank_shift, corr, scale_summary, scale_shift = compute_layer_tables(layer, args.rank_metric)
        summary.to_csv(os.path.join(analysis_dir, "layer_rank_summary.csv"), index=False)
        rank_shift.to_csv(os.path.join(analysis_dir, "layer_rank_shift.csv"), index=False)
        corr.to_csv(os.path.join(analysis_dir, "layer_rank_correlation.csv"), index=False)
        scale_summary.to_csv(os.path.join(analysis_dir, "layer_scale_summary.csv"), index=False)
        scale_shift.to_csv(os.path.join(analysis_dir, "layer_scale_shift.csv"), index=False)
        plot_layer_values(summary, plot_dir)
        print("\n=== Layer rank correlation ===")
        print(corr.to_string(index=False) if not corr.empty else "No layer correlation rows")
        print(f"\nSaved layer plots to {plot_dir}")

    print(f"\nSaved analysis files to {analysis_dir}")


if __name__ == "__main__":
    main()
