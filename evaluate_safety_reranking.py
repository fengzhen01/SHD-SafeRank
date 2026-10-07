from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from shd_saferank.data import load_prepared_data, make_fold_split
from shd_saferank.model import InductiveMultiTaskSafeRank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate decision-level efficacy-risk reranking"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--minimum-candidates", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    return parser.parse_args()


def tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype, device=torch.device("cpu"))


def build_model(data, config: dict) -> InductiveMultiTaskSafeRank:
    return InductiveMultiTaskSafeRank(
        herb_target_features=tensor(data.herb_target_features, torch.float32),
        herb_component_features=tensor(
            data.herb_component_features, torch.float32
        ),
        herb_feature_mask=tensor(data.herb_feature_mask, torch.float32),
        symptom_features=tensor(data.symptom_features, torch.float32),
        drug_features=tensor(data.drug_features, torch.float32),
        drug_feature_mask=tensor(data.drug_feature_mask, torch.float32),
        drug_target_features=tensor(data.drug_target_features, torch.float32),
        drug_target_feature_mask=tensor(
            data.drug_target_feature_mask, torch.float32
        ),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config.get("dropout", 0.2)),
        herb_architecture=str(config.get("herb_architecture", "shared_private")),
    )


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span <= 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - values.min()) / span


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, k: int) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return {"ap": np.nan, "mrr": np.nan, f"recall@{k}": np.nan}
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    first = int(np.flatnonzero(ranked == 1)[0])
    return {
        "ap": float(average_precision_score(labels, scores)),
        "mrr": float(1.0 / (first + 1)),
        f"recall@{k}": float(ranked[:k].sum() / positives),
    }


@torch.no_grad()
def evaluate_partition(
    model: InductiveMultiTaskSafeRank,
    data,
    hs_indices: np.ndarray,
    hdi_indices: np.ndarray,
    fold: int,
    lambdas: list[float],
    minimum_candidates: int,
    partition: str,
) -> list[dict]:
    model.eval()
    hs_test = data.hs_edges[hs_indices]
    hdi_test = data.hdi_edges[hdi_indices]
    hdi_labels = data.hdi_labels[hdi_indices]

    efficacy_herbs: dict[int, set[int]] = defaultdict(set)
    for herb, symptom in hs_test:
        efficacy_herbs[int(symptom)].add(int(herb))

    candidates_by_drug: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (herb, drug), label in zip(hdi_test, hdi_labels):
        candidates_by_drug[int(drug)].append((int(herb), int(label)))

    rows: list[dict] = []
    for symptom, positive_herbs in efficacy_herbs.items():
        for drug, candidate_pairs in candidates_by_drug.items():
            if len(candidate_pairs) < minimum_candidates:
                continue
            candidate_herbs = np.asarray(
                [pair[0] for pair in candidate_pairs], dtype=np.int64
            )
            observed_risk = np.asarray(
                [pair[1] for pair in candidate_pairs], dtype=np.int64
            )
            efficacy_labels = np.asarray(
                [int(herb in positive_herbs) for herb in candidate_herbs],
                dtype=np.int64,
            )
            if efficacy_labels.sum() == 0:
                continue

            herb_tensor = tensor(candidate_herbs, torch.long)
            symptom_tensor = torch.full(
                (len(candidate_herbs),), symptom, dtype=torch.long
            )
            drug_tensor = torch.full(
                (len(candidate_herbs),), drug, dtype=torch.long
            )
            efficacy_score = (
                model.score_herb_symptom(herb_tensor, symptom_tensor)
                .cpu()
                .numpy()
            )
            risk_probability = torch.sigmoid(
                model.score_herb_drug_risk(herb_tensor, drug_tensor)
            ).cpu().numpy()
            normalized_efficacy = minmax(efficacy_score)
            safe_efficacy = efficacy_labels * (1 - observed_risk)
            risky_efficacy = efficacy_labels * observed_risk

            for penalty in lambdas:
                utility = normalized_efficacy - float(penalty) * risk_probability
                order = np.argsort(-utility, kind="mergesort")
                top3 = order[: min(3, len(order))]
                efficacy = ranking_metrics(efficacy_labels, utility, 3)
                safe = ranking_metrics(safe_efficacy, utility, 3)
                risky = ranking_metrics(risky_efficacy, utility, 3)
                rows.append(
                    {
                        "fold": fold + 1,
                        "partition": partition,
                        "lambda": float(penalty),
                        "symptom_index": symptom,
                        "drug_index": drug,
                        "candidate_count": len(candidate_herbs),
                        "efficacy_positive_count": int(efficacy_labels.sum()),
                        "safe_efficacy_count": int(safe_efficacy.sum()),
                        "risky_efficacy_count": int(risky_efficacy.sum()),
                        "efficacy_ap": efficacy["ap"],
                        "efficacy_mrr": efficacy["mrr"],
                        "efficacy_recall@3": efficacy["recall@3"],
                        "safe_ap": safe["ap"],
                        "safe_mrr": safe["mrr"],
                        "safe_recall@3": safe["recall@3"],
                        "risky_efficacy_exposure@3": risky["recall@3"],
                        "top3_observed_risk_rate": float(
                            observed_risk[top3].mean()
                        ),
                        "top3_predicted_risk_mean": float(
                            risk_probability[top3].mean()
                        ),
                    }
                )
    return rows


def summarize(query_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "efficacy_ap",
        "efficacy_mrr",
        "efficacy_recall@3",
        "safe_ap",
        "safe_mrr",
        "safe_recall@3",
        "risky_efficacy_exposure@3",
        "top3_observed_risk_rate",
        "top3_predicted_risk_mean",
    ]
    fold_rows: list[dict] = []
    for (fold, penalty), group in query_frame.groupby(["fold", "lambda"]):
        row = {
            "fold": int(fold),
            "lambda": float(penalty),
            "queries": len(group),
            "safe_queries": int((group["safe_efficacy_count"] > 0).sum()),
            "risky_queries": int((group["risky_efficacy_count"] > 0).sum()),
        }
        row.update({metric: float(group[metric].mean()) for metric in metrics})
        fold_rows.append(row)
    by_fold = pd.DataFrame(fold_rows).sort_values(["lambda", "fold"])

    summary_rows: list[dict] = []
    for penalty, group in query_frame.groupby("lambda"):
        row = {
            "lambda": float(penalty),
            "queries": int(len(group)),
            "safe_queries": int((group["safe_efficacy_count"] > 0).sum()),
            "risky_queries": int((group["risky_efficacy_count"] > 0).sum()),
            "median_candidates": float(group["candidate_count"].median()),
        }
        row.update({metric: float(group[metric].mean()) for metric in metrics})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("lambda")
    return by_fold, summary


def select_penalty(
    validation_summary: pd.DataFrame, efficacy_tolerance: float = 0.01
) -> tuple[float, pd.DataFrame]:
    zero_rows = validation_summary[np.isclose(validation_summary["lambda"], 0.0)]
    if zero_rows.empty:
        raise ValueError("The lambda grid must include 0 for validation selection")
    baseline = float(zero_rows.iloc[0]["efficacy_ap"])
    eligible = validation_summary[
        validation_summary["efficacy_ap"] >= baseline - efficacy_tolerance
    ].copy()
    if eligible.empty:
        raise RuntimeError("No risk penalty satisfies the efficacy constraint")
    selected = eligible.sort_values(
        ["top3_observed_risk_rate", "lambda"], ascending=[True, True]
    ).iloc[0]
    validation_summary = validation_summary.copy()
    validation_summary["efficacy_constraint"] = (
        validation_summary["efficacy_ap"] >= baseline - efficacy_tolerance
    )
    validation_summary["selected"] = np.isclose(
        validation_summary["lambda"], float(selected["lambda"])
    )
    return float(selected["lambda"]), validation_summary


def paired_bootstrap(
    query_frame: pd.DataFrame,
    selected_lambda: float,
    samples: int,
    seed: int = 2026,
) -> pd.DataFrame:
    key_columns = ["fold", "symptom_index", "drug_index"]
    baseline = query_frame[np.isclose(query_frame["lambda"], 0.0)].set_index(
        key_columns
    )
    selected = query_frame[
        np.isclose(query_frame["lambda"], selected_lambda)
    ].set_index(key_columns)
    common = baseline.index.intersection(selected.index)
    comparisons = [
        ("efficacy_ap_gain", "efficacy_ap", 1.0),
        ("safe_ap_gain", "safe_ap", 1.0),
        (
            "top3_observed_risk_rate_reduction",
            "top3_observed_risk_rate",
            -1.0,
        ),
        (
            "risky_efficacy_exposure@3_reduction",
            "risky_efficacy_exposure@3",
            -1.0,
        ),
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for name, metric, direction in comparisons:
        base = baseline.loc[common, metric].to_numpy(dtype=float)
        chosen = selected.loc[common, metric].to_numpy(dtype=float)
        valid = np.isfinite(base) & np.isfinite(chosen)
        differences = direction * (chosen[valid] - base[valid])
        if len(differences) == 0:
            continue
        indices = rng.integers(0, len(differences), size=(samples, len(differences)))
        bootstrap_means = differences[indices].mean(axis=1)
        rows.append(
            {
                "comparison": name,
                "query_pairs": len(differences),
                "point_estimate": float(differences.mean()),
                "ci_2.5%": float(np.quantile(bootstrap_means, 0.025)),
                "ci_97.5%": float(np.quantile(bootstrap_means, 0.975)),
                "bootstrap_positive_fraction": float(
                    (bootstrap_means > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_tradeoff(by_fold: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("efficacy_ap", "Efficacy Macro-AP", "#2B6CB0"),
        ("safe_ap", "Safe-efficacy Macro-AP", "#2F855A"),
        ("top3_observed_risk_rate", "Observed risk rate@3", "#C53030"),
        (
            "risky_efficacy_exposure@3",
            "Risky-efficacy exposure@3",
            "#DD6B20",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for metric, label, color in metrics[:2]:
        grouped = by_fold.groupby("lambda")[metric]
        mean = grouped.mean()
        std = grouped.std(ddof=1).fillna(0.0)
        axes[0].errorbar(
            mean.index,
            mean.values,
            yerr=std.values,
            marker="o",
            capsize=3,
            label=label,
            color=color,
        )
    for metric, label, color in metrics[2:]:
        grouped = by_fold.groupby("lambda")[metric]
        mean = grouped.mean()
        std = grouped.std(ddof=1).fillna(0.0)
        axes[1].errorbar(
            mean.index,
            mean.values,
            yerr=std.values,
            marker="o",
            capsize=3,
            label=label,
            color=color,
        )
    axes[0].set_title("Efficacy retention")
    axes[1].set_title("Risk reduction")
    for axis in axes:
        axis.set_xlabel("Risk penalty (lambda)")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Metric value")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "inductive":
        raise ValueError("Safety reranking requires an inductive model config")
    if config.get("split_strategy") != "cold_herb":
        raise ValueError(
            "Safety reranking currently requires a cold_herb experiment so "
            "both binary test relations refer to held-out herbs"
        )

    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    result_dir = Path(config["output_dir"])
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_prepared_data(data_path)
    validation_rows: list[dict] = []
    test_rows: list[dict] = []
    for fold in range(int(config["folds"])):
        split = make_fold_split(
            data=data,
            fold=fold,
            n_splits=int(config["folds"]),
            seed=int(config["seed"]),
            validation_fraction=float(config["validation_fraction"]),
            strategy="cold_herb",
        )
        model = build_model(data, config)
        checkpoint = torch.load(
            result_dir / "checkpoints" / f"fold_{fold + 1}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        validation_rows.extend(
            evaluate_partition(
                model=model,
                data=data,
                hs_indices=split.hs_val,
                hdi_indices=split.hdi_val,
                fold=fold,
                lambdas=list(args.lambdas),
                minimum_candidates=int(args.minimum_candidates),
                partition="validation",
            )
        )
        test_rows.extend(
            evaluate_partition(
                model=model,
                data=data,
                hs_indices=split.hs_test,
                hdi_indices=split.hdi_test,
                fold=fold,
                lambdas=list(args.lambdas),
                minimum_candidates=int(args.minimum_candidates),
                partition="test",
            )
        )

    validation_frame = pd.DataFrame(validation_rows)
    query_frame = pd.DataFrame(test_rows)
    if validation_frame.empty:
        raise RuntimeError("No eligible validation symptom-drug queries were constructed")
    if query_frame.empty:
        raise RuntimeError("No eligible symptom-drug queries were constructed")
    _, validation_summary = summarize(validation_frame)
    selected_lambda, validation_summary = select_penalty(validation_summary)
    by_fold, summary = summarize(query_frame)
    selected_test = query_frame[
        np.isclose(query_frame["lambda"], selected_lambda)
    ].copy()
    _, selected_summary = summarize(selected_test)
    validation_frame.to_csv(
        output_dir / "validation_query_metrics.csv", index=False
    )
    validation_summary.to_csv(
        output_dir / "validation_tradeoff_summary.csv", index=False
    )
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    by_fold.to_csv(output_dir / "tradeoff_by_fold.csv", index=False)
    summary.to_csv(output_dir / "tradeoff_summary.csv", index=False)
    selected_test.to_csv(output_dir / "selected_test_query_metrics.csv", index=False)
    selected_summary.to_csv(output_dir / "selected_test_summary.csv", index=False)
    bootstrap = paired_bootstrap(
        query_frame=query_frame,
        selected_lambda=selected_lambda,
        samples=int(args.bootstrap_samples),
    )
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    plot_tradeoff(by_fold, output_dir / "safety_tradeoff.png")
    metadata = {
        "source_config": str(config_path),
        "source_results": str(result_dir),
        "lambdas": list(args.lambdas),
        "minimum_candidates": int(args.minimum_candidates),
        "bootstrap_samples": int(args.bootstrap_samples),
        "selected_lambda": selected_lambda,
        "selection_rule": (
            "Using pooled validation queries, require efficacy Macro-AP to be "
            "within 0.01 absolute of lambda=0, then minimize observed risk "
            "rate@3; ties prefer the smaller lambda."
        ),
        "evaluation_semantics": (
            "Decision-level composition of held-out herb-symptom efficacy "
            "edges and held-out herb-drug risk labels. It is not direct "
            "ternary clinical supervision."
        ),
        "candidate_boundary": (
            "Each symptom-drug query ranks only held-out herbs with a known "
            "HDI test label for that drug."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validation-selected lambda: {selected_lambda:g}")
    print("Selected test result:")
    print(selected_summary.to_string(index=False))
    print("Paired query bootstrap (positive values favor reranking):")
    print(bootstrap.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
