from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_safety_reranking import build_model
from run_experiment import risk_probabilities
from shd_saferank.data import load_prepared_data, make_fold_split
from shd_saferank.metrics import binary_metrics, full_candidate_hs_metrics
from paper_plotting import save_figure_bundle


FEATURE_GROUPS = {
    "herb_target": ("herb_target_features", {"HS", "HDI"}),
    "herb_component": ("herb_component_features", {"HS", "HDI"}),
    "symptom_gene": ("symptom_features", {"HS"}),
    "drug_molecular": ("drug_features", {"HDI"}),
    "drug_target": ("drug_target_features", {"HDI"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature-group and gate analysis for SHD-SafeRank"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    return parser.parse_args()


@contextmanager
def zero_buffer(model, buffer_name: str):
    buffer = getattr(model, buffer_name)
    original = buffer.detach().clone()
    with torch.no_grad():
        buffer.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            buffer.copy_(original)


def evaluate_tasks(model, data, split, threshold: float) -> dict:
    device = torch.device("cpu")
    hs = full_candidate_hs_metrics(
        model=model,
        test_edges=data.hs_edges[split.hs_test],
        train_edges=data.hs_edges[split.hs_train],
        candidate_herbs=np.flatnonzero(data.herb_feature_mask[:, 0] > 0.5),
        device=device,
    )
    hdi_edges = data.hdi_edges[split.hdi_test]
    probabilities = risk_probabilities(model, hdi_edges, device)
    hdi = binary_metrics(
        data.hdi_labels[split.hdi_test], probabilities, threshold
    )
    return {
        "hs_macro_ap": hs["macro_ap"],
        "hs_mrr": hs["mrr"],
        "hdi_auroc": hdi["auroc"],
        "hdi_auprc": hdi["auprc"],
        "hdi_mcc": hdi["mcc"],
        "hdi_probabilities": probabilities,
    }


@torch.no_grad()
def gate_rows(model, data, split, fold: int) -> list[dict]:
    test_herbs = np.unique(
        np.concatenate(
            [
                data.hs_edges[split.hs_test, 0],
                data.hdi_edges[split.hdi_test, 0],
            ]
        )
    )
    herb_tensor = torch.as_tensor(test_herbs, dtype=torch.long)
    inputs = model._herb_input(herb_tensor)
    shared = model.shared_herb_encoder(inputs)
    efficacy_private = model.efficacy_herb_encoder(inputs)
    risk_private = model.risk_herb_encoder(inputs)
    efficacy_gate = torch.sigmoid(
        model.efficacy_gate(torch.cat([shared, efficacy_private], dim=-1))
    )
    risk_gate = torch.sigmoid(
        model.risk_gate(torch.cat([shared, risk_private], dim=-1))
    )
    rows: list[dict] = []
    for position, herb in enumerate(test_herbs):
        rows.append(
            {
                "fold": fold + 1,
                "herb_index": int(herb),
                "efficacy_shared_gate_mean": float(
                    efficacy_gate[position].mean().item()
                ),
                "efficacy_shared_gate_std": float(
                    efficacy_gate[position].std().item()
                ),
                "risk_shared_gate_mean": float(risk_gate[position].mean().item()),
                "risk_shared_gate_std": float(risk_gate[position].std().item()),
            }
        )
    return rows


@torch.no_grad()
def benchmark(model, data, repeats: int) -> dict:
    rng = np.random.default_rng(2026)
    batch_size = 10000
    herbs = torch.as_tensor(
        rng.integers(0, data.n_herbs, size=batch_size), dtype=torch.long
    )
    symptoms = torch.as_tensor(
        rng.integers(0, data.n_symptoms, size=batch_size), dtype=torch.long
    )
    drugs = torch.as_tensor(
        rng.integers(0, data.n_drugs, size=batch_size), dtype=torch.long
    )
    model.eval()
    for _ in range(5):
        model.score_herb_symptom(herbs, symptoms)
        model.score_herb_drug_risk(herbs, drugs)
    hs_start = time.perf_counter()
    for _ in range(repeats):
        model.score_herb_symptom(herbs, symptoms)
    hs_seconds = time.perf_counter() - hs_start
    hdi_start = time.perf_counter()
    for _ in range(repeats):
        model.score_herb_drug_risk(herbs, drugs)
    hdi_seconds = time.perf_counter() - hdi_start
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "parameter_count": parameters,
        "trainable_parameter_count": trainable,
        "parameter_memory_mb_float32": parameters * 4 / (1024**2),
        "benchmark_batch_size": batch_size,
        "benchmark_repeats": repeats,
        "hs_pairs_per_second": batch_size * repeats / hs_seconds,
        "hdi_pairs_per_second": batch_size * repeats / hdi_seconds,
        "hs_milliseconds_per_10000_pairs": hs_seconds / repeats * 1000,
        "hdi_milliseconds_per_10000_pairs": hdi_seconds / repeats * 1000,
    }


def plot_results(
    ablation: pd.DataFrame,
    gates: pd.DataFrame,
    risks: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)

    plot_rows = []
    for (group, task), frame in ablation.groupby(["feature_group", "task"]):
        metric = "macro_ap_drop" if task == "HS" else "auprc_drop"
        plot_rows.append(
            {
                "label": f"{group}\n({task})",
                "mean": frame[metric].mean(),
                "std": frame[metric].std(ddof=1),
            }
        )
    plot_frame = pd.DataFrame(plot_rows).sort_values("mean", ascending=False)
    axes[0].bar(
        np.arange(len(plot_frame)),
        plot_frame["mean"],
        yerr=plot_frame["std"].fillna(0.0),
        capsize=3,
        color="#4C78A8",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(
        np.arange(len(plot_frame)), plot_frame["label"], rotation=35, ha="right"
    )
    axes[0].set_ylabel("Performance decrease")
    axes[0].set_title("Feature-group zero ablation")
    axes[0].set_yscale("symlog", linthresh=0.005)
    axes[0].grid(axis="y", alpha=0.25)
    for position, row in plot_frame.reset_index(drop=True).iterrows():
        axes[0].text(
            position,
            row["mean"],
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom" if row["mean"] >= 0 else "top",
            fontsize=7,
        )

    axes[1].boxplot(
        [
            gates["efficacy_shared_gate_mean"],
            gates["risk_shared_gate_mean"],
        ],
        labels=["Efficacy gate", "Risk gate"],
        patch_artist=True,
        boxprops={"facecolor": "#72B7B2"},
        medianprops={"color": "black"},
    )
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Shared-representation weight")
    axes[1].set_title("Task-specific herb gates")
    axes[1].grid(axis="y", alpha=0.25)

    nonrisk = risks.loc[risks["risk_label"] == 0, "risk_probability"]
    risk = risks.loc[risks["risk_label"] == 1, "risk_probability"]
    axes[2].boxplot(
        [nonrisk, risk],
        labels=["No-effect only", "Harmful/Negative"],
        patch_artist=True,
        boxprops={"facecolor": "#F58518"},
        medianprops={"color": "black"},
    )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Predicted risk probability")
    axes[2].set_title("Held-out HDI score distribution")
    axes[2].grid(axis="y", alpha=0.25)
    save_figure_bundle(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output = args.output if args.output.is_absolute() else project_root / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "inductive":
        raise ValueError("Explainability analysis requires an inductive model")
    if config.get("herb_architecture", "shared_private") != "shared_private":
        raise ValueError("Gate analysis requires the shared-private model")
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    result_dir = Path(config["output_dir"])
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    data = load_prepared_data(data_path)
    fold_metrics = pd.read_csv(result_dir / "fold_metrics.csv").set_index("fold")

    ablation_rows: list[dict] = []
    gate_output: list[dict] = []
    risk_rows: list[dict] = []
    first_model = None
    for fold in range(int(config["folds"])):
        split = make_fold_split(
            data=data,
            fold=fold,
            n_splits=int(config["folds"]),
            seed=int(config["seed"]),
            validation_fraction=float(config["validation_fraction"]),
            strategy=str(config["split_strategy"]),
        )
        model = build_model(data, config)
        checkpoint_path = result_dir / "checkpoints" / f"fold_{fold + 1}.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        if first_model is None:
            first_model = model
        threshold = float(fold_metrics.loc[fold + 1, "hdi_threshold"])
        baseline = evaluate_tasks(model, data, split, threshold)
        hdi_test = data.hdi_edges[split.hdi_test]
        for edge, label, probability in zip(
            hdi_test,
            data.hdi_labels[split.hdi_test],
            baseline["hdi_probabilities"],
        ):
            risk_rows.append(
                {
                    "fold": fold + 1,
                    "herb_index": int(edge[0]),
                    "drug_index": int(edge[1]),
                    "risk_label": int(label),
                    "risk_probability": float(probability),
                }
            )
        gate_output.extend(gate_rows(model, data, split, fold))

        for feature_group, (buffer_name, tasks) in FEATURE_GROUPS.items():
            with zero_buffer(model, buffer_name):
                ablated = evaluate_tasks(model, data, split, threshold)
            if "HS" in tasks:
                ablation_rows.append(
                    {
                        "fold": fold + 1,
                        "feature_group": feature_group,
                        "task": "HS",
                        "baseline_macro_ap": baseline["hs_macro_ap"],
                        "ablated_macro_ap": ablated["hs_macro_ap"],
                        "macro_ap_drop": (
                            baseline["hs_macro_ap"] - ablated["hs_macro_ap"]
                        ),
                        "baseline_auprc": np.nan,
                        "ablated_auprc": np.nan,
                        "auprc_drop": np.nan,
                    }
                )
            if "HDI" in tasks:
                ablation_rows.append(
                    {
                        "fold": fold + 1,
                        "feature_group": feature_group,
                        "task": "HDI",
                        "baseline_macro_ap": np.nan,
                        "ablated_macro_ap": np.nan,
                        "macro_ap_drop": np.nan,
                        "baseline_auprc": baseline["hdi_auprc"],
                        "ablated_auprc": ablated["hdi_auprc"],
                        "auprc_drop": (
                            baseline["hdi_auprc"] - ablated["hdi_auprc"]
                        ),
                    }
                )

    ablation_frame = pd.DataFrame(ablation_rows)
    gate_frame = pd.DataFrame(gate_output)
    risk_frame = pd.DataFrame(risk_rows)
    ablation_frame.to_csv(output / "feature_group_ablation.csv", index=False)
    gate_frame.to_csv(output / "gate_values.csv", index=False)
    risk_frame.to_csv(output / "risk_probability_distribution.csv", index=False)
    summary_rows = []
    for (feature_group, task), frame in ablation_frame.groupby(
        ["feature_group", "task"]
    ):
        metric = "macro_ap_drop" if task == "HS" else "auprc_drop"
        summary_rows.append(
            {
                "feature_group": feature_group,
                "task": task,
                "metric": "Macro-AP" if task == "HS" else "AUPRC",
                "mean_drop": float(frame[metric].mean()),
                "std_drop": float(frame[metric].std(ddof=1)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output / "feature_group_ablation_summary.csv", index=False
    )
    gate_summary = gate_frame[
        ["efficacy_shared_gate_mean", "risk_shared_gate_mean"]
    ].agg(["mean", "std", "median", "min", "max"])
    gate_summary.to_csv(output / "gate_summary.csv")
    if first_model is None:
        raise RuntimeError("No checkpoint was loaded")
    efficiency = benchmark(first_model, data, int(args.benchmark_repeats))
    efficiency["checkpoint_size_mb"] = (
        (result_dir / "checkpoints" / "fold_1.pt").stat().st_size / (1024**2)
    )
    (output / "cpu_efficiency.json").write_text(
        json.dumps(efficiency, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(
        ablation_frame,
        gate_frame,
        risk_frame,
        output / "explainability_overview.png",
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(gate_summary.to_string())
    print(json.dumps(efficiency, indent=2))


if __name__ == "__main__":
    main()
