# SIREN Independent EN/KO/FR Language-Rank Experiment

This package runs the graduation-project experiment with **independent datasets per language**.
It does **not** translate English/Korean/French samples. Each language is evaluated on three
language-specific safety/toxicity/hate/offensiveness datasets.

## Dataset design

The config avoids the common SIREN benchmark/training-family datasets such as ToxicChat,
OpenAIMod, Aegis/Aegis2, WildGuard, SafeRLHF, BeaverTails, and the previously used Korean
Ko-WildGuard / Ko-PKU-SafeRLHF training data.

Default selected datasets:

| Language | Dataset key | Source |
|---|---|---|
| English | `en_toxigen` | `toxigen/toxigen-data` |
| English | `en_civil_comments` | `google/civil_comments` |
| English | `en_hatecheck` | `Paul/hatecheck` |
| Korean | `ko_kmhas` | `jeanlee/kmhas_korean_hate_speech` |
| Korean | `ko_kor_hate` | `moon1ite/kor_hate` |
| Korean | `ko_kold` | `nayohan/KOLD` |
| French | `fr_mlma` | `nedjmaou/MLMA_hate_speech` |
| French | `fr_hatecheck` | `Paul/hatecheck-french` |
| French | `fr_camer_hate_fr` | local CSV from Mendeley Camer-Hate-FR |

For `fr_camer_hate_fr`, download the original French CSV from Mendeley and place it at:

```bash
data/camer_hate_fr_dataset.csv
```

Use the original French file, not the English translations.

## Install

```bash
pip install -r requirements.txt
```

Log in to Hugging Face if needed for Llama/Qwen backbones:

```bash
huggingface-cli login
```

## Configure

Edit `configs/exp_lang_rank.yaml`:

```yaml
models:
  - name: Ko-SIREN-Qwen3-4B-local
    kind: pkl_siren
    pkl_path: /path/to/best_model.pkl
    base_model: /path/to/Qwen3-4B
```

You can set the per-dataset cap here:

```yaml
data:
  default_max_samples_per_dataset: 1000
```

Set it to `null` for full datasets.

## Run

```bash
bash scripts/run_exp_lang_rank.sh configs/exp_lang_rank.yaml
```

Outputs:

```text
outputs/lang_rank_exp/
  lang_eval.jsonl
  predictions.jsonl
  layer_values.csv
  analysis/
    metrics.csv
    consistency.csv                 # aggregate language-level score/prediction-rate shift
    layer_rank_summary.csv
    layer_rank_shift.csv            # language-level rank/value shift
    layer_rank_correlation.csv      # Spearman correlation between EN/KO/FR layer-rank profiles
    plots/
```

## Notes

Because the datasets are independent across languages, exact sample-level flip-rate is no longer
meaningful. The analyzer instead reports:

1. dataset-level metrics for each language dataset,
2. aggregate language-level unsafe prediction-rate and mean-score shifts,
3. layer-rank profiles and Spearman rank correlations between language-level profiles.

This is a better fit for the revised design: language robustness is measured as distributional
robustness across independent native-language benchmarks rather than invariance under translation.

## Layer-rank and layer-scale interpretation

This code does **not** rank layers by the frozen backbone model's parameter weights. That would not match SIREN.

For each SIREN model, safety neurons are already selected from the trained SIREN artifact:

```text
selected_neuron_activation[layer] = hidden_state[layer][selected_neuron_indices]
weighted_feature[layer] = selected_neuron_activation[layer] * fixed_layer_weight[layer]
```

The rank/scale analysis uses these diagnostic values:

```text
raw_abs_sum        = sum(abs(selected_neuron_activation))
weighted_abs_sum   = sum(abs(selected_neuron_activation * layer_weight))
weighted_abs_share = weighted_abs_sum / sum_all_layers(weighted_abs_sum)
```

`weighted_abs_sum` is a useful pre-MLP contribution proxy, but it is **not** an exact causal attribution of the final MLP score.
The output now includes:

```text
layer_scale_summary.csv
layer_scale_shift.csv
```

`layer_scale_shift.csv` includes log-ratio changes such as:

```text
weighted_abs_log_ratio = log((target_weighted_abs_sum + eps) / (base_weighted_abs_sum + eps))
```
