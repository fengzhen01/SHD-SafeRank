from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_safety_reranking import build_model, minmax
from shd_saferank.data import load_prepared_data, make_fold_split
from paper_plotting import save_figure_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and visualize successful and challenging reranking cases"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reranking-results", type=Path, required=True)
    parser.add_argument("--processed-data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-candidates", type=int, default=5)
    return parser.parse_args()


def choose_cases(query_frame: pd.DataFrame, selected_lambda: float, minimum: int):
    keys = ["fold", "symptom_index", "drug_index"]
    base = query_frame[np.isclose(query_frame["lambda"], 0.0)].set_index(keys)
    selected = query_frame[
        np.isclose(query_frame["lambda"], selected_lambda)
    ].set_index(keys)
    common = base.index.intersection(selected.index)
    comparison = pd.DataFrame(index=common)
    comparison["candidate_count"] = base.loc[common, "candidate_count"]
    comparison["safe_efficacy_count"] = base.loc[common, "safe_efficacy_count"]
    comparison["risky_efficacy_count"] = base.loc[common, "risky_efficacy_count"]
    comparison["efficacy_ap_change"] = (
        selected.loc[common, "efficacy_ap"] - base.loc[common, "efficacy_ap"]
    )
    comparison["safe_ap_change"] = (
        selected.loc[common, "safe_ap"] - base.loc[common, "safe_ap"]
    )
    comparison["risk_rate_reduction"] = (
        base.loc[common, "top3_observed_risk_rate"]
        - selected.loc[common, "top3_observed_risk_rate"]
    )
    comparison = comparison[comparison["candidate_count"] >= minimum]
    if comparison.empty:
        raise RuntimeError("No queries satisfy the case-selection candidate limit")

    success_pool = comparison[
        (comparison["risk_rate_reduction"] > 0)
        & (comparison["efficacy_ap_change"] >= -0.01)
        & (comparison["safe_efficacy_count"] > 0)
    ]
    if success_pool.empty:
        success_pool = comparison[
            (comparison["risk_rate_reduction"] > 0)
            & (comparison["safe_efficacy_count"] > 0)
        ]
    success = success_pool.sort_values(
        ["risk_rate_reduction", "efficacy_ap_change"],
        ascending=[False, False],
    ).iloc[0]

    difficult_pool = comparison[comparison["risk_rate_reduction"] <= 0]
    if difficult_pool.empty:
        difficult_pool = comparison
    difficult = difficult_pool.sort_values(
        ["efficacy_ap_change", "risk_rate_reduction"],
        ascending=[True, True],
    ).iloc[0]
    return (success.name, success), (difficult.name, difficult)


def load_name_maps(processed_dir: Path, data) -> tuple[dict, dict]:
    crosswalk = pd.read_csv(processed_dir / "herb_crosswalk.csv")
    herb_map: dict[int, dict] = {}
    by_key = crosswalk.drop_duplicates("unified_herb_key").set_index(
        "unified_herb_key"
    )
    for index, key in enumerate(data.herb_keys):
        key_text = str(key)
        if key_text in by_key.index:
            row = by_key.loc[key_text]
            herb_map[index] = {
                "herb_key": key_text,
                "hit_tcm_id": str(row.get("hit_tcm_id", "")),
                "hdi_herb_id": str(row.get("hdi_herb_id", "")),
                "herb_name": str(row.get("hit_chinese_name", "")),
            }
        else:
            herb_map[index] = {
                "herb_key": key_text,
                "hit_tcm_id": "",
                "hdi_herb_id": "",
                "herb_name": "",
            }
    weak_paths = pd.read_csv(processed_dir / "weak_paths.csv")
    drug_map = (
        weak_paths[["Drug_ID", "Drug_Name"]]
        .drop_duplicates("Drug_ID")
        .set_index("Drug_ID")["Drug_Name"]
        .astype(str)
        .to_dict()
    )
    return herb_map, drug_map


@torch.no_grad()
def candidate_details(
    model,
    data,
    split,
    fold: int,
    symptom: int,
    drug: int,
    penalty: float,
    case_type: str,
    herb_map: dict,
    drug_map: dict,
) -> pd.DataFrame:
    hs_test = data.hs_edges[split.hs_test]
    hdi_test = data.hdi_edges[split.hdi_test]
    hdi_labels = data.hdi_labels[split.hdi_test]
    positive_herbs = {
        int(herb) for herb, symptom_index in hs_test if int(symptom_index) == symptom
    }
    candidates = [
        (int(edge[0]), int(label))
        for edge, label in zip(hdi_test, hdi_labels)
        if int(edge[1]) == drug
    ]
    candidate_herbs = np.asarray([item[0] for item in candidates], dtype=np.int64)
    observed_risk = np.asarray([item[1] for item in candidates], dtype=np.int64)
    herb_tensor = torch.as_tensor(candidate_herbs, dtype=torch.long)
    symptom_tensor = torch.full((len(candidates),), symptom, dtype=torch.long)
    drug_tensor = torch.full((len(candidates),), drug, dtype=torch.long)
    efficacy_raw = model.score_herb_symptom(herb_tensor, symptom_tensor).numpy()
    efficacy = minmax(efficacy_raw)
    risk_probability = torch.sigmoid(
        model.score_herb_drug_risk(herb_tensor, drug_tensor)
    ).numpy()
    utility = efficacy - penalty * risk_probability
    baseline_order = np.argsort(-efficacy, kind="mergesort")
    reranked_order = np.argsort(-utility, kind="mergesort")
    baseline_rank = np.empty(len(candidates), dtype=int)
    reranked_rank = np.empty(len(candidates), dtype=int)
    baseline_rank[baseline_order] = np.arange(1, len(candidates) + 1)
    reranked_rank[reranked_order] = np.arange(1, len(candidates) + 1)
    rows = []
    drug_id = str(data.drug_ids[drug])
    for position, herb in enumerate(candidate_herbs):
        names = herb_map[int(herb)]
        display_id = names["hit_tcm_id"]
        if not display_id or display_id == "nan":
            display_id = names["hdi_herb_id"]
        if not display_id or display_id == "nan":
            display_id = f"H{int(herb)}"
        rows.append(
            {
                "case_type": case_type,
                "fold": fold,
                "symptom_index": symptom,
                "symptom_umls_id": str(data.symptom_ids[symptom]),
                "drug_index": drug,
                "drug_id": drug_id,
                "drug_name": drug_map.get(drug_id, ""),
                "herb_index": int(herb),
                "herb_display_id": display_id,
                "herb_name": names["herb_name"],
                "herb_key": names["herb_key"],
                "efficacy_relevant": int(int(herb) in positive_herbs),
                "observed_risk": int(observed_risk[position]),
                "normalized_efficacy_score": float(efficacy[position]),
                "predicted_risk_probability": float(risk_probability[position]),
                "reranking_utility": float(utility[position]),
                "baseline_rank": int(baseline_rank[position]),
                "reranked_rank": int(reranked_rank[position]),
                "rank_change": int(baseline_rank[position] - reranked_rank[position]),
            }
        )
    return pd.DataFrame(rows)


def plot_cases(details: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases = ["successful", "challenging"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for axis, case_type in zip(axes, cases):
        frame = details[details["case_type"] == case_type].copy()
        frame["best_rank"] = frame[["baseline_rank", "reranked_rank"]].min(axis=1)
        frame = frame.sort_values("best_rank").head(12)
        maximum_rank = int(
            frame[["baseline_rank", "reranked_rank"]].to_numpy().max()
        )
        axis.plot(
            [0.5, maximum_rank + 0.5],
            [0.5, maximum_rank + 0.5],
            linestyle="--",
            color="gray",
            linewidth=1,
        )
        for _, row in frame.iterrows():
            color = "#C53030" if int(row["observed_risk"]) == 1 else "#2F855A"
            marker = "*" if int(row["efficacy_relevant"]) == 1 else "o"
            size = 150 if int(row["efficacy_relevant"]) == 1 else 55
            axis.scatter(
                row["baseline_rank"],
                row["reranked_rank"],
                marker=marker,
                s=size,
                color=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.7,
            )
            axis.annotate(
                str(row["herb_display_id"]),
                (row["baseline_rank"], row["reranked_rank"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        first = frame.iloc[0]
        axis.set_xlim(0.5, maximum_rank + 0.5)
        axis.set_ylim(0.5, maximum_rank + 0.5)
        axis.set_xlabel("Efficacy-only rank")
        axis.set_ylabel("Safety-aware rank")
        axis.invert_xaxis()
        axis.invert_yaxis()
        axis.grid(alpha=0.2)
        axis.set_title(
            f"{case_type.title()} case\n"
            f"{first['symptom_umls_id']} + {first['drug_id']} {first['drug_name']}"
        )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2F855A", label="No-effect-only evidence"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C53030", label="Harmful/Negative evidence"),
        Line2D([0], [0], marker="*", markersize=12, color="gray", linestyle="none", label="Held-out efficacy association"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.20, top=0.82, wspace=0.28)
    save_figure_bundle(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    rerank_dir = (
        args.reranking_results
        if args.reranking_results.is_absolute()
        else project_root / args.reranking_results
    )
    processed = (
        args.processed_data
        if args.processed_data.is_absolute()
        else project_root / args.processed_data
    )
    output = args.output if args.output.is_absolute() else project_root / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads((rerank_dir / "metadata.json").read_text(encoding="utf-8"))
    selected_lambda = float(metadata["selected_lambda"])
    query_frame = pd.read_csv(rerank_dir / "query_metrics.csv")
    success, difficult = choose_cases(
        query_frame, selected_lambda, int(args.minimum_candidates)
    )
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    result_dir = Path(config["output_dir"])
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    data = load_prepared_data(data_path)
    herb_map, drug_map = load_name_maps(processed, data)

    detail_frames = []
    summaries = []
    for case_type, (key, row) in (
        ("successful", success),
        ("challenging", difficult),
    ):
        fold, symptom, drug = map(int, key)
        split = make_fold_split(
            data=data,
            fold=fold - 1,
            n_splits=int(config["folds"]),
            seed=int(config["seed"]),
            validation_fraction=float(config["validation_fraction"]),
            strategy=str(config["split_strategy"]),
        )
        model = build_model(data, config)
        checkpoint = torch.load(
            result_dir / "checkpoints" / f"fold_{fold}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        details = candidate_details(
            model=model,
            data=data,
            split=split,
            fold=fold,
            symptom=symptom,
            drug=drug,
            penalty=selected_lambda,
            case_type=case_type,
            herb_map=herb_map,
            drug_map=drug_map,
        )
        detail_frames.append(details)
        first = details.iloc[0]
        summaries.append(
            {
                "case_type": case_type,
                "selection_rule": (
                    "maximum risk-rate reduction with efficacy AP loss <=0.01"
                    if case_type == "successful"
                    else "largest efficacy AP loss among queries without risk-rate improvement"
                ),
                "fold": fold,
                "symptom_umls_id": first["symptom_umls_id"],
                "drug_id": first["drug_id"],
                "drug_name": first["drug_name"],
                "candidate_count": int(row["candidate_count"]),
                "efficacy_ap_change": float(row["efficacy_ap_change"]),
                "safe_ap_change": float(row["safe_ap_change"])
                if np.isfinite(row["safe_ap_change"])
                else np.nan,
                "risk_rate@3_reduction": float(row["risk_rate_reduction"]),
                "selected_lambda": selected_lambda,
            }
        )
    detail_frame = pd.concat(detail_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    detail_frame.to_csv(output / "case_candidate_rankings.csv", index=False)
    summary_frame.to_csv(output / "case_summary.csv", index=False)
    plot_cases(detail_frame, output / "case_rank_changes.png")
    print(summary_frame.to_string(index=False))
    for case_type in ("successful", "challenging"):
        print(f"\n{case_type.upper()} TOP RERANKED")
        columns = [
            "herb_display_id",
            "herb_name",
            "efficacy_relevant",
            "observed_risk",
            "baseline_rank",
            "reranked_rank",
            "predicted_risk_probability",
        ]
        print(
            detail_frame[detail_frame["case_type"] == case_type]
            .sort_values("reranked_rank")
            .head(10)[columns]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
