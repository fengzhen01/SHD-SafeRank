from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "final_strict_symptom_query"
OUT.mkdir(parents=True, exist_ok=True)


def load(run: str) -> dict[str, float]:
    payload = json.loads((RESULTS / run / "summary.json").read_text(encoding="utf-8"))
    return {key: value["mean"] for key, value in payload["summary"].items()}


metrics = ["hs_macro_ap", "hs_mrr", "hs_recall@5", "hs_recall@10", "hs_ndcg@10", "hdi_auroc", "hdi_auprc", "hdi_f1", "hdi_mcc"]


rows = []
for split, seeds in [("edge", [0, 17, 42]), ("cold_herb", [0, 17, 42]), ("cold_drug", [0]), ("cold_both", [0, 17, 42])]:
    values = [load(f"strict_symptom_query_{split}_seed{seed}") for seed in seeds]
    row = {"protocol": split, "seeds": len(seeds)}
    for metric in metrics:
        array = np.asarray([item[metric] for item in values], dtype=float)
        row[f"{metric}_mean"] = float(array.mean())
        row[f"{metric}_std_across_seeds"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "protocol_aggregate.csv", index=False)


rows = []
for split in ["edge", "cold_herb", "cold_drug", "cold_both"]:
    for encoder, run in [("ID", f"strict_symptom_query_id_{split}_seed0"), ("Inductive", f"strict_symptom_query_{split}_seed0")]:
        values = load(run)
        rows.append({"split": split, "encoder": encoder, **{m: values[m] for m in metrics}})
pd.DataFrame(rows).to_csv(OUT / "id_vs_inductive.csv", index=False)


control_runs = [
    ("Joint inductive", "strict_symptom_query_edge_seed0"),
    ("HS single-task", "strict_symptom_query_inductive_hs_only_edge_seed0"),
    ("HDI single-task", "strict_symptom_query_inductive_hdi_only_edge_seed0"),
    ("Drug-ID only", "strict_symptom_query_id_drug_only_edge_seed0"),
    ("Herb-ID only", "strict_symptom_query_id_herb_only_edge_seed0"),
    ("Permuted HDI labels", "strict_symptom_query_id_permuted_edge_seed0"),
]
rows = []
for label, run in control_runs:
    values = load(run)
    rows.append({"setting": label, **{m: values[m] for m in metrics}})
pd.DataFrame(rows).to_csv(OUT / "diagnostic_controls.csv", index=False)


rows = []
for architecture, run in [
    ("shared_private", "strict_symptom_query_cold_herb_seed0"),
    ("shared_only", "strict_symptom_query_shared_only_cold_herb_seed0"),
    ("private_only", "strict_symptom_query_private_only_cold_herb_seed0"),
]:
    values = load(run)
    rows.append({"architecture": architecture, **{m: values[m] for m in metrics}})
pd.DataFrame(rows).to_csv(OUT / "architecture_ablation.csv", index=False)


rows = []
for split, seeds in [("cold_drug", [0]), ("cold_both", [0, 17, 42])]:
    for status, prefix in [("With DrugBank targets", "strict_symptom_query"), ("Without DrugBank targets", "strict_symptom_query_no_drug_target")]:
        values = [load(f"{prefix}_{split}_seed{seed}") for seed in seeds]
        for metric in ["hdi_auroc", "hdi_auprc", "hdi_mcc"]:
            array = np.asarray([item[metric] for item in values], dtype=float)
            rows.append({"split": split, "target_status": status, "metric": metric, "mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0, "n_seeds": len(array)})
pd.DataFrame(rows).to_csv(OUT / "drug_target_ablation.csv", index=False)


pd.read_csv(RESULTS / "strict_symptom_query_safety_multi_seed" / "multi_seed_results.csv").to_csv(OUT / "safety_reranking_multi_seed.csv", index=False)

manifest = {
    "canonical_protocol": {
        "hs_query_direction": "symptom_to_herb",
        "candidate_herbs": 798,
        "hs_negative_samples_per_positive": 5,
        "hs_negative_filter_scope": "train_only",
        "cross_validation_folds": 5,
        "formal_random_seeds": [0, 17, 42],
    },
    "old_all_known_results": "Sensitivity/audit only; excluded from main manuscript tables.",
    "authoritative_tables": sorted(path.name for path in OUT.glob("*.csv")),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(OUT)
