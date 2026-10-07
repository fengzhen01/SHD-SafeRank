from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paper_plotting import save_figure_bundle

from shd_saferank.data import load_prepared_data, make_fold_split
from shd_saferank.metrics import (
    binary_metrics,
    full_candidate_hs_metrics,
    select_mcc_threshold,
)


class TargetCosineModel(torch.nn.Module):
    def __init__(self, herb_features: np.ndarray, symptom_features: np.ndarray):
        super().__init__()
        self.register_buffer(
            "herb", F.normalize(torch.as_tensor(herb_features).float(), dim=-1)
        )
        self.register_buffer(
            "symptom",
            F.normalize(torch.as_tensor(symptom_features).float(), dim=-1),
        )

    def score_herb_symptom(self, herbs, symptoms):
        return (self.herb[herbs] * self.symptom[symptoms]).sum(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-aware traditional baselines")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pair_features(data, edges: np.ndarray) -> np.ndarray:
    herbs = edges[:, 0]
    drugs = edges[:, 1]
    return np.concatenate(
        [
            data.herb_target_features[herbs],
            data.herb_component_features[herbs],
            data.herb_feature_mask[herbs],
            data.drug_features[drugs],
            data.drug_target_features[drugs],
            data.drug_feature_mask[drugs, None],
            data.drug_target_feature_mask[drugs, None],
        ],
        axis=1,
    ).astype(np.float32)


def target_overlap_scores(data, edges: np.ndarray) -> np.ndarray:
    herb = data.herb_target_features[edges[:, 0]]
    drug = data.drug_target_features[edges[:, 1]]
    herb_norm = np.linalg.norm(herb, axis=1, keepdims=True)
    drug_norm = np.linalg.norm(drug, axis=1, keepdims=True)
    herb = herb / np.maximum(herb_norm, 1e-12)
    drug = drug / np.maximum(drug_norm, 1e-12)
    cosine = np.sum(herb * drug, axis=1)
    return np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method"):
        row = {"method": method, "folds": len(group)}
        for metric in ["hs_macro_ap", "hs_mrr", "hdi_auroc", "hdi_auprc", "hdi_mcc"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, main_metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs_row = summary[summary["method"] == "Target cosine (HS)"].iloc[0]
    overlap_row = summary[summary["method"] == "Target cosine (HDI)"].iloc[0]
    logistic_row = summary[summary["method"] == "Logistic regression (HDI)"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].bar(
        [0, 1],
        [hs_row["hs_macro_ap_mean"], main_metrics["hs_macro_ap"].mean()],
        yerr=[hs_row["hs_macro_ap_std"], main_metrics["hs_macro_ap"].std(ddof=1)],
        capsize=4,
        color=["#A0AEC0", "#2B6CB0"],
    )
    axes[0].set_xticks([0, 1], ["Target cosine", "Shared-private model"])
    axes[0].set_ylabel("Macro-AP")
    axes[0].set_title("Herb-symptom ranking")
    axes[0].set_ylim(0, 0.65)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        [0, 1, 2],
        [
            overlap_row["hdi_auprc_mean"],
            logistic_row["hdi_auprc_mean"],
            main_metrics["hdi_auprc"].mean(),
        ],
        yerr=[
            overlap_row["hdi_auprc_std"],
            logistic_row["hdi_auprc_std"],
            main_metrics["hdi_auprc"].std(ddof=1),
        ],
        capsize=4,
        color=["#A0AEC0", "#72B7B2", "#F58518"],
    )
    axes[1].set_xticks(
        [0, 1, 2], ["Target cosine", "Logistic", "Shared-private"]
    )
    axes[1].set_ylabel("AUPRC")
    axes[1].set_title("Herb-drug risk prediction")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.25)
    save_figure_bundle(fig, output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output = args.output if args.output.is_absolute() else project_root / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    result_dir = Path(config["output_dir"])
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    data = load_prepared_data(data_path)
    cosine_model = TargetCosineModel(
        data.herb_target_features, data.symptom_features
    )
    rows = []
    for fold in range(int(config["folds"])):
        split = make_fold_split(
            data=data,
            fold=fold,
            n_splits=int(config["folds"]),
            seed=int(config["seed"]),
            validation_fraction=float(config["validation_fraction"]),
            strategy=str(config["split_strategy"]),
        )
        hs = full_candidate_hs_metrics(
            model=cosine_model,
            test_edges=data.hs_edges[split.hs_test],
            train_edges=data.hs_edges[split.hs_train],
            candidate_herbs=np.flatnonzero(data.herb_feature_mask[:, 0] > 0.5),
            device=torch.device("cpu"),
        )
        rows.append(
            {
                "fold": fold + 1,
                "method": "Target cosine (HS)",
                "hs_macro_ap": hs["macro_ap"],
                "hs_mrr": hs["mrr"],
                "hdi_auroc": np.nan,
                "hdi_auprc": np.nan,
                "hdi_mcc": np.nan,
            }
        )

        train_edges = data.hdi_edges[split.hdi_train]
        val_edges = data.hdi_edges[split.hdi_val]
        test_edges = data.hdi_edges[split.hdi_test]
        y_train = data.hdi_labels[split.hdi_train]
        y_val = data.hdi_labels[split.hdi_val]
        y_test = data.hdi_labels[split.hdi_test]

        overlap_val = target_overlap_scores(data, val_edges)
        overlap_test = target_overlap_scores(data, test_edges)
        overlap_threshold = select_mcc_threshold(y_val, overlap_val)
        overlap_metrics = binary_metrics(y_test, overlap_test, overlap_threshold)
        rows.append(
            {
                "fold": fold + 1,
                "method": "Target cosine (HDI)",
                "hs_macro_ap": np.nan,
                "hs_mrr": np.nan,
                "hdi_auroc": overlap_metrics["auroc"],
                "hdi_auprc": overlap_metrics["auprc"],
                "hdi_mcc": overlap_metrics["mcc"],
            }
        )

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=int(config["seed"]) + fold,
            ),
        )
        classifier.fit(pair_features(data, train_edges), y_train)
        logistic_val = classifier.predict_proba(pair_features(data, val_edges))[:, 1]
        logistic_test = classifier.predict_proba(pair_features(data, test_edges))[:, 1]
        logistic_threshold = select_mcc_threshold(y_val, logistic_val)
        logistic_metrics = binary_metrics(y_test, logistic_test, logistic_threshold)
        rows.append(
            {
                "fold": fold + 1,
                "method": "Logistic regression (HDI)",
                "hs_macro_ap": np.nan,
                "hs_mrr": np.nan,
                "hdi_auroc": logistic_metrics["auroc"],
                "hdi_auprc": logistic_metrics["auprc"],
                "hdi_mcc": logistic_metrics["mcc"],
            }
        )

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    frame.to_csv(output / "traditional_baseline_folds.csv", index=False)
    summary.to_csv(output / "traditional_baseline_summary.csv", index=False)
    main_metrics = pd.read_csv(result_dir / "fold_metrics.csv")
    plot(summary, main_metrics, output / "traditional_baseline_comparison.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
