# SHD-SafeRank

SHD-SafeRank is a leakage-aware, CPU-friendly research framework for **symptom-conditioned herb ranking** and **herb–drug interaction (HDI) risk discrimination**. It learns the two binary evidence tasks with a shared–private inductive encoder and combines them at decision time through validation-selected safety-aware reranking.

> This repository supports computational knowledge discovery and candidate prioritization. It does **not** provide clinical diagnosis, prescribing, interaction checking, or medication advice.

## Why this project

Most herb recommendation studies optimize association accuracy without explicitly considering potential risks when herbal medicines and conventional drugs are used together. SHD-SafeRank addresses this gap while enforcing auditable evaluation boundaries:

- symptom-to-herb ranking over all 798 HIT candidate herbs;
- pairwise BPR learning from positive–unlabeled herb–symptom data;
- evidence-weighted BCE for HDI risk evidence;
- shared–private inductive entity encoders;
- random-edge, cold-herb, cold-drug, and double-cold evaluation;
- validation-only selection of the reranking penalty;
- query-level paired bootstrap, feature ablation, and case analysis;
- no direct symptom–herb–drug ternary supervision.

## Evidence boundary

Symptom–herb–drug paths are created by joining two observed binary relations through a shared herb. They are exploratory composed paths, **not independently validated clinical ternary facts**. The model therefore learns:

1. a herb–symptom association score; and
2. a herb–drug risk probability.

The final utility is computed only at decision time:

```text
utility(herb | symptom, drug) = association_score - lambda * risk_probability
```

`lambda` is selected on validation queries and remains fixed for test evaluation.

## Data summary

| Data component | Scale | Role |
|---|---:|---|
| HIT herb–symptom | 798 herbs, 174 symptoms, 1,436 unique associations | full-candidate ranking |
| HDI risk evidence | 738 risk pairs | supervised positive class |
| HDI No Effect evidence | 404 pairs | non-risk reference class |
| Other HDI evidence | 6,366 pairs | unlabeled; excluded from supervised risk loss |
| Cross-task herbs | 68 | shared entity analysis |

The compact processed dataset is included in `data/processed/`. See [DATA.md](DATA.md) for provenance, label definitions, and redistribution boundaries.

## Repository layout

```text
SHD-SafeRank/
├── shd_saferank/                  # model, data splits, losses, and metrics
├── configs/strict_train_only/     # canonical leakage-aware configurations
├── data/processed/                # compact processed arrays and metadata
├── results/final_strict_symptom_query/
│   └── *.csv                      # authoritative aggregate result tables
├── figures/final_rerun/           # PNG, editable SVG, and vector PDF figures
├── tests/                         # protocol and implementation tests
├── prepare_data.py                # rebuild processed data from source databases
├── run_experiment.py              # train/evaluate one configuration
└── run_strict_train_only_suite.py # reproduce the complete strict experiment suite
```

## Installation

Conda is recommended for the CPU environment:

```bash
conda env create -f environment-cpu.yml
conda activate shd-saferank-cpu
```

Alternatively:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run one strict five-fold experiment:

```bash
python run_experiment.py --config configs/strict_train_only/symptom_query_edge_seed0.json
```

Run the full canonical suite:

```bash
python run_strict_train_only_suite.py
python run_strict_post_analyses.py
python summarize_strict_final.py
```

The canonical protocol holds the symptom fixed, ranks all 798 HIT herbs, samples five unlabeled herbs per positive association, and builds the negative-sampling exclusion set from **training-fold positives only**.

## Main strict results

Mean ± standard deviation is reported across seeds 0, 17, and 42 unless noted otherwise.

| Protocol | HS Macro-AP | HDI AUROC | HDI AUPRC | HDI MCC |
|---|---:|---:|---:|---:|
| Random edge | 0.0602±0.0087 | 0.9755±0.0048 | 0.9844±0.0034 | 0.8269±0.0147 |
| Cold herb | 0.0179±0.0003 | 0.9317±0.0088 | 0.9478±0.0065 | 0.6802±0.0520 |
| Cold drug (seed 0) | 0.0449 | 0.6119 | 0.7938 | 0.2129 |
| Double cold | 0.0125±0.0005 | 0.5803±0.1203 | 0.7319±0.0120 | 0.1233±0.1673 |

Safety-aware reranking reduced the observed Top-3 risk rate in all three seeds, with a mean absolute reduction of `0.0147±0.0095`. Efficacy ranking changes were seed-dependent (`-0.0183±0.0516`), so the method should be interpreted as an auditable efficacy–risk trade-off rather than a universally improving reranker.

## Reproducibility notes

- Duplicate binary relations are removed before splitting.
- Held-out positives are not used to construct the training negative pool.
- Validation and test labels are excluded from model fitting.
- HDI thresholds and reranking penalties are selected on validation data only.
- Pre-generated herb–symptom embeddings that may encode target edges are excluded.
- Potential label-proxy fields from the HDI evidence table are not model inputs.
- Older `all_known` and reverse-query results are audit artifacts and are not the canonical paper results.

## Citation and data use

If you use this repository, cite the original HIT/MHGNN, HDI, and DrugBank-derived data sources described in [DATA.md](DATA.md), as well as the accompanying SHD-SafeRank manuscript when available. No open-source software license has been selected yet; contact the repository owner before redistributing modified code.
