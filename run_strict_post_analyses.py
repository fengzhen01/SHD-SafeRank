from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT / "configs" / "strict_train_only"
RESULTS = ROOT / "results"


def run(args: list[str]) -> None:
    print(" ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    safety_dirs = []
    for seed in [0, 17, 42]:
        config = CONFIGS / f"symptom_query_cold_herb_seed{seed}.json"
        output = RESULTS / f"strict_symptom_query_safety_cold_herb_seed{seed}"
        run(
            [
                "evaluate_safety_reranking.py",
                "--config",
                str(config),
                "--output",
                str(output),
                "--lambdas",
                "0",
                "0.25",
                "0.5",
                "0.75",
                "1",
                "1.5",
                "2",
                "3",
                "5",
                "10",
            ]
        )
        safety_dirs.append(output)
    run(
        [
            "aggregate_safety_reranking.py",
            "--inputs",
            *[str(path) for path in safety_dirs],
            "--output",
            str(RESULTS / "strict_symptom_query_safety_multi_seed"),
        ]
    )
    config0 = CONFIGS / "symptom_query_cold_herb_seed0.json"
    run(
        [
            "analyze_explainability.py",
            "--config",
            str(config0),
            "--output",
            str(RESULTS / "strict_symptom_query_explainability_cold_herb_seed0"),
        ]
    )
    run(
        [
            "analyze_cases.py",
            "--config",
            str(config0),
            "--reranking-results",
            str(safety_dirs[0]),
            "--processed-data",
            "data/processed",
            "--output",
            str(RESULTS / "strict_symptom_query_case_analysis_seed0"),
        ]
    )
    run(
        [
            "run_traditional_baselines.py",
            "--config",
            str(config0),
            "--output",
            str(RESULTS / "strict_symptom_query_traditional_baselines_cold_herb_seed0"),
        ]
    )


if __name__ == "__main__":
    main()
