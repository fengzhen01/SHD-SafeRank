from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paper_plotting import save_figure_bundle


METRICS = [
    "efficacy_ap",
    "safe_ap",
    "top3_observed_risk_rate",
    "risky_efficacy_exposure@3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed safety reranking results"
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot infer seed from directory: {path}")
    return int(match.group(1))


def load_seed_result(path: Path) -> dict:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(path / "tradeoff_summary.csv")
    baseline = curve[np.isclose(curve["lambda"], 0.0)].iloc[0]
    selected_lambda = float(metadata["selected_lambda"])
    selected = curve[np.isclose(curve["lambda"], selected_lambda)].iloc[0]
    row = {
        "seed": seed_from_path(path),
        "selected_lambda": selected_lambda,
        "queries": int(selected["queries"]),
        "safe_queries": int(selected["safe_queries"]),
        "risky_queries": int(selected["risky_queries"]),
    }
    for metric in METRICS:
        row[f"baseline_{metric}"] = float(baseline[metric])
        row[f"selected_{metric}"] = float(selected[metric])
    row["efficacy_ap_change"] = (
        row["selected_efficacy_ap"] - row["baseline_efficacy_ap"]
    )
    row["safe_ap_change"] = row["selected_safe_ap"] - row["baseline_safe_ap"]
    row["observed_risk_rate@3_reduction"] = (
        row["baseline_top3_observed_risk_rate"]
        - row["selected_top3_observed_risk_rate"]
    )
    row["risky_efficacy_exposure@3_reduction"] = (
        row["baseline_risky_efficacy_exposure@3"]
        - row["selected_risky_efficacy_exposure@3"]
    )
    return row


def plot(seed_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Efficacy\nMacro-AP",
        "Safe-efficacy\nMacro-AP",
        "Observed risk\nrate@3",
        "Risky-efficacy\nexposure@3",
    ]
    baseline_columns = [f"baseline_{metric}" for metric in METRICS]
    selected_columns = [f"selected_{metric}" for metric in METRICS]
    baseline_mean = seed_frame[baseline_columns].mean().to_numpy()
    baseline_std = seed_frame[baseline_columns].std(ddof=1).to_numpy()
    selected_mean = seed_frame[selected_columns].mean().to_numpy()
    selected_std = seed_frame[selected_columns].std(ddof=1).to_numpy()

    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    axis.bar(
        x - width / 2,
        baseline_mean,
        width,
        yerr=baseline_std,
        capsize=4,
        label="Efficacy only (lambda=0)",
        color="#4C78A8",
    )
    axis.bar(
        x + width / 2,
        selected_mean,
        width,
        yerr=selected_std,
        capsize=4,
        label="Validation-selected safety reranking",
        color="#F58518",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 0.85)
    axis.set_ylabel("Metric value")
    axis.set_title("Multi-seed efficacy-risk trade-off")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    save_figure_bundle(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    inputs = [path if path.is_absolute() else project_root / path for path in args.inputs]
    output = args.output if args.output.is_absolute() else project_root / args.output
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([load_seed_result(path) for path in inputs]).sort_values(
        "seed"
    )
    frame.to_csv(output / "multi_seed_results.csv", index=False)
    change_columns = [
        "efficacy_ap_change",
        "safe_ap_change",
        "observed_risk_rate@3_reduction",
        "risky_efficacy_exposure@3_reduction",
    ]
    aggregate = {
        column: {
            "mean": float(frame[column].mean()),
            "std_across_seeds": float(frame[column].std(ddof=1)),
        }
        for column in change_columns
    }
    aggregate["seeds"] = frame["seed"].astype(int).tolist()
    aggregate["selected_lambdas"] = frame["selected_lambda"].tolist()
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(frame, output / "multi_seed_safety_tradeoff.png")
    print(frame.to_string(index=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
