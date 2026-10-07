from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from shd_saferank.data import (
    known_herbs_by_symptom,
    load_prepared_data,
    make_fold_split,
    sample_hs_negative_herbs,
    split_audit,
)
from shd_saferank.metrics import (
    binary_metrics,
    full_candidate_hs_metrics,
    select_mcc_threshold,
)
from shd_saferank.model import InductiveMultiTaskSafeRank, MultiTaskSafeRank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHD-SafeRank")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def tensor(values: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype, device=device)


def bpr_loss(
    model: MultiTaskSafeRank,
    symptoms: np.ndarray,
    positive_herbs: np.ndarray,
    negative_herbs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    s = tensor(symptoms, torch.long, device)
    p = tensor(positive_herbs, torch.long, device)
    n = tensor(negative_herbs, torch.long, device)
    positive_scores = model.score_herb_symptom(p, s)
    negative_scores = model.score_herb_symptom(n, s)
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def risk_loss(
    model: MultiTaskSafeRank,
    edges: np.ndarray,
    labels: np.ndarray,
    evidence_weights: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    edge_tensor = tensor(edges, torch.long, device)
    y = tensor(labels, torch.float32, device)
    evidence = tensor(evidence_weights, torch.float32, device)
    logits = model.score_herb_drug_risk(edge_tensor[:, 0], edge_tensor[:, 1])
    positives = max(1, int(labels.sum()))
    negatives = max(1, int(len(labels) - labels.sum()))
    positive_weight = len(labels) / (2.0 * positives)
    negative_weight = len(labels) / (2.0 * negatives)
    class_weights = torch.where(
        y > 0.5,
        torch.full_like(y, positive_weight),
        torch.full_like(y, negative_weight),
    )
    losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (losses * class_weights * evidence).sum() / (
        class_weights * evidence
    ).sum().clamp_min(1e-8)


@torch.no_grad()
def risk_probabilities(
    model: MultiTaskSafeRank, edges: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    edge_tensor = tensor(edges, torch.long, device)
    logits = model.score_herb_drug_risk(edge_tensor[:, 0], edge_tensor[:, 1])
    return torch.sigmoid(logits).cpu().numpy()


def train_fold(config: dict, data, fold: int, output_dir: Path) -> dict:
    seed = int(config["seed"]) + fold
    set_seed(seed)
    device = torch.device("cpu")
    split = make_fold_split(
        data=data,
        fold=fold,
        n_splits=int(config["folds"]),
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
        strategy=str(config.get("split_strategy", "edge")),
    )
    audit = split_audit(
        data, split, strategy=str(config.get("split_strategy", "edge"))
    )
    overlap_fields = [
        "hs_train_val_overlap",
        "hs_train_test_overlap",
        "hs_val_test_overlap",
        "hdi_train_val_overlap",
        "hdi_train_test_overlap",
        "hdi_val_test_overlap",
    ]
    split_strategy = config.get("split_strategy", "edge")
    if split_strategy == "cold_herb":
        overlap_fields.extend(
            [
                "cold_herb_train_val_overlap",
                "cold_herb_train_test_overlap",
                "cold_herb_val_test_overlap",
            ]
        )
    elif split_strategy == "cold_both":
        overlap_fields.extend(
            [
                "cold_herb_train_test_overlap",
                "cold_herb_val_test_overlap",
            ]
        )
    if split_strategy == "cold_drug":
        overlap_fields.extend(
            [
                "cold_drug_train_val_overlap",
                "cold_drug_train_test_overlap",
                "cold_drug_val_test_overlap",
            ]
        )
    elif split_strategy == "cold_both":
        overlap_fields.extend(
            [
                "cold_drug_train_test_overlap",
                "cold_drug_val_test_overlap",
            ]
        )
    if any(audit[key] != 0 for key in overlap_fields):
        raise RuntimeError(f"Split leakage detected: {audit}")
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "splits" / f"fold_{fold + 1}_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    if str(config.get("model_type", "id_embedding")) == "inductive":
        use_drug_target_features = bool(
            config.get("use_drug_target_features", True)
        )
        drug_target_features = (
            data.drug_target_features
            if use_drug_target_features
            else np.zeros_like(data.drug_target_features)
        )
        drug_target_feature_mask = (
            data.drug_target_feature_mask
            if use_drug_target_features
            else np.zeros_like(data.drug_target_feature_mask)
        )
        model = InductiveMultiTaskSafeRank(
            herb_target_features=tensor(
                data.herb_target_features, torch.float32, device
            ),
            herb_component_features=tensor(
                data.herb_component_features, torch.float32, device
            ),
            herb_feature_mask=tensor(
                data.herb_feature_mask, torch.float32, device
            ),
            symptom_features=tensor(data.symptom_features, torch.float32, device),
            drug_features=tensor(data.drug_features, torch.float32, device),
            drug_feature_mask=tensor(
                data.drug_feature_mask, torch.float32, device
            ),
            drug_target_features=tensor(
                drug_target_features, torch.float32, device
            ),
            drug_target_feature_mask=tensor(
                drug_target_feature_mask, torch.float32, device
            ),
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config.get("dropout", 0.2)),
            herb_architecture=str(
                config.get("herb_architecture", "shared_private")
            ),
        ).to(device)
    else:
        model = MultiTaskSafeRank(
            n_herbs=data.n_herbs,
            n_symptoms=data.n_symptoms,
            n_drugs=data.n_drugs,
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            hdi_feature_mode=str(config.get("hdi_feature_mode", "both")),
        ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    hs_train_edges = data.hs_edges[split.hs_train]
    hs_val_edges = data.hs_edges[split.hs_val]
    hs_test_edges = data.hs_edges[split.hs_test]
    hdi_train_edges = data.hdi_edges[split.hdi_train]
    hdi_val_edges = data.hdi_edges[split.hdi_val]
    hdi_test_edges = data.hdi_edges[split.hdi_test]
    hdi_train_labels = data.hdi_labels[split.hdi_train].copy()
    hdi_train_weights = data.hdi_weights[split.hdi_train].copy()
    hdi_val_labels = data.hdi_labels[split.hdi_val].copy()
    hdi_val_weights = data.hdi_weights[split.hdi_val].copy()
    negative_filter_scope = str(
        config.get("hs_negative_filter_scope", "all_known")
    )
    if negative_filter_scope == "all_known":
        negative_filter_edges = data.hs_edges
    elif negative_filter_scope == "train_only":
        negative_filter_edges = hs_train_edges
    else:
        raise ValueError(
            "hs_negative_filter_scope must be 'all_known' or 'train_only'"
        )
    known_hs = known_herbs_by_symptom(negative_filter_edges)
    hs_candidate_herbs = np.flatnonzero(data.herb_feature_mask[:, 0] > 0.5)
    if len(hs_candidate_herbs) == 0:
        raise RuntimeError("No HIT herb candidates were identified")
    if bool(config.get("permute_hdi_train_labels", False)):
        permutation_rng = np.random.default_rng(seed + 30000)
        permutation = permutation_rng.permutation(len(hdi_train_labels))
        hdi_train_labels = hdi_train_labels[permutation]
        hdi_train_weights = hdi_train_weights[permutation]
        validation_permutation = permutation_rng.permutation(len(hdi_val_labels))
        hdi_val_labels = hdi_val_labels[validation_permutation]
        hdi_val_weights = hdi_val_weights[validation_permutation]

    validation_rng = np.random.default_rng(seed + 20000)
    val_s, val_p, val_n = sample_hs_negative_herbs(
        hs_val_edges,
        known_hs,
        hs_candidate_herbs,
        int(config["hs_negatives"]),
        validation_rng,
    )
    best_state = None
    best_validation_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        rng = np.random.default_rng(seed * 100000 + epoch)
        train_s, train_p, train_n = sample_hs_negative_herbs(
            hs_train_edges,
            known_hs,
            hs_candidate_herbs,
            int(config["hs_negatives"]),
            rng,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_hs = bpr_loss(model, train_s, train_p, train_n, device)
        loss_hdi = risk_loss(
            model,
            hdi_train_edges,
            hdi_train_labels,
            hdi_train_weights,
            device,
        )
        total_loss = (
            float(config.get("hs_loss_weight", 1.0)) * loss_hs
            + float(config["hdi_loss_weight"]) * loss_hdi
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_hs = bpr_loss(model, val_s, val_p, val_n, device)
            validation_hdi = risk_loss(
                model,
                hdi_val_edges,
                hdi_val_labels,
                hdi_val_weights,
                device,
            )
            validation_loss = float(
                float(config.get("hs_loss_weight", 1.0)) * validation_hs.item()
                + float(config["hdi_loss_weight"]) * validation_hdi.item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_hs_loss": float(loss_hs.item()),
                "train_hdi_loss": float(loss_hdi.item()),
                "train_total_loss": float(total_loss.item()),
                "validation_hs_loss": float(validation_hs.item()),
                "validation_hdi_loss": float(validation_hdi.item()),
                "validation_total_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= int(config["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "fold": fold,
            "config": config,
            "data_dimensions": {
                "herbs": data.n_herbs,
                "symptoms": data.n_symptoms,
                "drugs": data.n_drugs,
            },
        },
        output_dir / "checkpoints" / f"fold_{fold + 1}.pt",
    )
    pd.DataFrame(history).to_csv(
        output_dir / f"history_fold_{fold + 1}.csv", index=False
    )

    hs_metrics = full_candidate_hs_metrics(
        model=model,
        test_edges=hs_test_edges,
        train_edges=hs_train_edges,
        candidate_herbs=hs_candidate_herbs,
        device=device,
    )
    validation_probabilities = risk_probabilities(model, hdi_val_edges, device)
    threshold = select_mcc_threshold(
        hdi_val_labels, validation_probabilities
    )
    test_probabilities = risk_probabilities(model, hdi_test_edges, device)
    hdi_metrics = binary_metrics(
        data.hdi_labels[split.hdi_test], test_probabilities, threshold
    )

    result = {
        "fold": fold + 1,
        "epochs_trained": len(history),
        "best_validation_loss": best_validation_loss,
        **{f"hs_{key}": value for key, value in hs_metrics.items()},
        **{f"hdi_{key}": value for key, value in hdi_metrics.items()},
    }
    print(json.dumps(result, indent=2))
    return result


def summarize(fold_frame: pd.DataFrame) -> dict:
    summary: dict[str, dict[str, float]] = {}
    numeric_columns = [
        column
        for column in fold_frame.columns
        if column not in {"fold"} and pd.api.types.is_numeric_dtype(fold_frame[column])
    ]
    for column in numeric_columns:
        summary[column] = {
            "mean": float(fold_frame[column].mean()),
            "std": float(fold_frame[column].std(ddof=1))
            if len(fold_frame) > 1
            else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = project_root / data_path
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_prepared_data(data_path)
    config = dict(config)
    config["data_path"] = str(data_path)
    config["output_dir"] = str(output_dir)
    config["device"] = "cpu"
    (output_dir / "config.resolved.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    requested = config.get("run_folds")
    folds = list(range(int(config["folds"]))) if requested is None else requested
    results = [train_fold(config, data, int(fold), output_dir) for fold in folds]
    fold_frame = pd.DataFrame(results)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    output = {
        "folds_completed": len(results),
        "fold_indices_zero_based": folds,
        "hs_query_direction": "symptom_to_herb",
        "hs_negative_filter_scope": config.get(
            "hs_negative_filter_scope", "all_known"
        ),
        "summary": summarize(fold_frame),
        "interpretation_boundary": (
            "The model evaluates two binary tasks. Composed symptom-herb-drug "
            "paths are not used as direct ternary supervision."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
