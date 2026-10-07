from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .data import known_herbs_by_symptom


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "threshold": float(threshold),
    }


def select_mcc_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_mcc = -math.inf
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(np.int64)
        mcc = matthews_corrcoef(y_true, predictions)
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_threshold = float(threshold)
    return best_threshold


def _ranking_for_query(
    scores: np.ndarray,
    positives: set[int],
    filtered: set[int],
    k_values=(1, 5, 10),
    item_ids: np.ndarray | None = None,
) -> dict:
    if item_ids is None:
        item_ids = np.arange(len(scores), dtype=np.int64)
    else:
        item_ids = np.asarray(item_ids, dtype=np.int64)
    if len(scores) != len(item_ids):
        raise ValueError("scores and item_ids must have identical lengths")
    excluded = filtered - positives
    candidate_mask = np.asarray(
        [int(item) not in excluded for item in item_ids], dtype=bool
    )
    candidates = item_ids[candidate_mask]
    candidate_scores = scores[candidate_mask]
    order = candidates[np.argsort(-candidate_scores, kind="mergesort")]
    relevance = np.asarray([1 if item in positives else 0 for item in order], dtype=int)
    cumulative = np.cumsum(relevance)
    positive_positions = np.flatnonzero(relevance == 1)
    if len(positive_positions) == 0:
        return {}
    precisions = cumulative[positive_positions] / (positive_positions + 1)
    result = {
        "ap": float(precisions.mean()),
        "mrr": float(1.0 / (positive_positions[0] + 1)),
    }
    for k in k_values:
        top = relevance[:k]
        result[f"recall@{k}"] = float(top.sum() / len(positives))
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float((top * discounts).sum())
        ideal_len = min(len(positives), k)
        ideal = float((1.0 / np.log2(np.arange(2, ideal_len + 2))).sum())
        result[f"ndcg@{k}"] = dcg / ideal if ideal > 0 else 0.0
    return result


@torch.no_grad()
def full_candidate_hs_metrics(
    model,
    test_edges: np.ndarray,
    train_edges: np.ndarray,
    candidate_herbs: np.ndarray,
    device: torch.device,
) -> dict:
    test_by_symptom = known_herbs_by_symptom(test_edges)
    train_by_symptom = known_herbs_by_symptom(train_edges)
    rows = []
    candidate_herbs = np.asarray(candidate_herbs, dtype=np.int64)
    herb_tensor = torch.as_tensor(candidate_herbs, dtype=torch.long, device=device)
    model.eval()
    for symptom, positives in test_by_symptom.items():
        symptom_tensor = torch.full(
            (len(candidate_herbs),), symptom, dtype=torch.long, device=device
        )
        scores = (
            model.score_herb_symptom(herb_tensor, symptom_tensor)
            .detach()
            .cpu()
            .numpy()
        )
        result = _ranking_for_query(
            scores,
            positives,
            train_by_symptom.get(symptom, set()),
            item_ids=candidate_herbs,
        )
        if result:
            rows.append(result)
    if not rows:
        return {
            "macro_ap": float("nan"),
            "mrr": float("nan"),
            "recall@1": float("nan"),
            "recall@5": float("nan"),
            "recall@10": float("nan"),
            "ndcg@1": float("nan"),
            "ndcg@5": float("nan"),
            "ndcg@10": float("nan"),
            "queries": 0,
        }
    keys = rows[0].keys()
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    aggregate["macro_ap"] = aggregate.pop("ap")
    aggregate["queries"] = len(rows)
    return aggregate
