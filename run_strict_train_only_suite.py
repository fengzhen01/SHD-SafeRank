from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "strict_train_only"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "results" / "strict_train_only_suite.log"


SOURCES = [
    "symptom_query_edge_seed0.json",
    "symptom_query_edge_seed17.json",
    "symptom_query_edge_seed42.json",
    "symptom_query_cold_herb_seed0.json",
    "symptom_query_cold_herb_seed17.json",
    "symptom_query_cold_herb_seed42.json",
    "symptom_query_cold_drug_seed0.json",
    "symptom_query_cold_both_seed0.json",
    "symptom_query_cold_both_seed17.json",
    "symptom_query_cold_both_seed42.json",
    "rerun_id_edge_seed0.json",
    "rerun_id_cold_herb_seed0.json",
    "rerun_id_cold_drug_seed0.json",
    "rerun_id_cold_both_seed0.json",
    "rerun_inductive_hs_only_edge_seed0.json",
    "rerun_inductive_hdi_only_edge_seed0.json",
    "rerun_id_drug_only_edge_seed0.json",
    "rerun_id_herb_only_edge_seed0.json",
    "rerun_id_permuted_edge_seed0.json",
    "symptom_query_shared_only_cold_herb_seed0.json",
    "symptom_query_private_only_cold_herb_seed0.json",
    "rerun_no_drug_target_cold_drug_seed0.json",
    "rerun_no_drug_target_cold_both_seed0.json",
    "rerun_no_drug_target_cold_both_seed17.json",
    "rerun_no_drug_target_cold_both_seed42.json",
]


def strict_name(output_dir: str) -> str:
    name = Path(output_dir).name
    prefix = "hit_candidate_symptom_query_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    return f"results/strict_symptom_query_{name}"


def main() -> None:
    start = time.perf_counter()
    completed = []
    with LOG.open("w", encoding="utf-8") as log:
        for position, source_name in enumerate(SOURCES, start=1):
            source_path = ROOT / "configs" / source_name
            config = json.loads(source_path.read_text(encoding="utf-8"))
            config["hs_negative_filter_scope"] = "train_only"
            config["output_dir"] = strict_name(config["output_dir"])
            target_path = CONFIG_DIR / source_name
            target_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{position:02d}/{len(SOURCES)}] {source_name}", flush=True)
            result = subprocess.run(
                [sys.executable, "run_experiment.py", "--config", str(target_path)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                raise SystemExit(f"Failed: {source_name}; inspect {LOG}")
            completed.append(config["output_dir"])
    elapsed = time.perf_counter() - start
    manifest = {
        "protocol": "symptom_to_herb_train_only",
        "completed_runs": completed,
        "elapsed_seconds": elapsed,
        "python": sys.executable,
        "log": str(LOG),
    }
    (ROOT / "results" / "strict_train_only_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed {len(completed)} runs in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
