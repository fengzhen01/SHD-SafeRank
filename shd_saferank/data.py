from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass(frozen=True)
class PreparedData:
    hs_edges: np.ndarray
    hdi_edges: np.ndarray
    hdi_labels: np.ndarray
    hdi_weights: np.ndarray
    hdi_unlabeled_edges: np.ndarray
    herb_keys: np.ndarray
    symptom_ids: np.ndarray
    drug_ids: np.ndarray
    herb_target_features: np.ndarray
    herb_component_features: np.ndarray
    herb_feature_mask: np.ndarray
    symptom_features: np.ndarray
    drug_features: np.ndarray
    drug_feature_mask: np.ndarray
    drug_target_features: np.ndarray
    drug_target_feature_mask: np.ndarray

    @property
    def n_herbs(self) -> int:
        return len(self.herb_keys)

    @property
    def n_symptoms(self) -> int:
        return len(self.symptom_ids)

    @property
    def n_drugs(self) -> int:
        return len(self.drug_ids)


@dataclass(frozen=True)
class FoldSplit:
    hs_train: np.ndarray
    hs_val: np.ndarray
    hs_test: np.ndarray
    hdi_train: np.ndarray
    hdi_val: np.ndarray
    hdi_test: np.ndarray


def load_prepared_data(path: str | Path) -> PreparedData:
    archive = np.load(Path(path), allow_pickle=False)
    return PreparedData(
        hs_edges=archive["hs_edges"].astype(np.int64),
        hdi_edges=archive["hdi_edges"].astype(np.int64),
        hdi_labels=archive["hdi_labels"].astype(np.int64),
        hdi_weights=archive["hdi_weights"].astype(np.float32),
        hdi_unlabeled_edges=archive["hdi_unlabeled_edges"].astype(np.int64),
        herb_keys=archive["unified_herb_keys"],
        symptom_ids=archive["symptom_ids"],
        drug_ids=archive["drug_ids"],
        herb_target_features=archive["herb_target_features"].astype(np.float32),
        herb_component_features=archive["herb_component_features"].astype(np.float32),
        herb_feature_mask=archive["herb_feature_mask"].astype(np.float32),
        symptom_features=archive["symptom_features"].astype(np.float32),
        drug_features=archive["drug_features"].astype(np.float32),
        drug_feature_mask=archive["drug_feature_mask"].astype(np.float32),
        drug_target_features=archive["drug_target_features"].astype(np.float32),
        drug_target_feature_mask=archive["drug_target_feature_mask"].astype(
            np.float32
        ),
    )


def make_fold_split(
    data: PreparedData,
    fold: int,
    n_splits: int,
    seed: int,
    validation_fraction: float,
    strategy: str = "edge",
) -> FoldSplit:
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits})")

    if strategy == "cold_herb":
        return _make_cold_herb_split(
            data=data,
            fold=fold,
            n_splits=n_splits,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    if strategy == "cold_drug":
        return _make_cold_drug_split(
            data=data,
            fold=fold,
            n_splits=n_splits,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    if strategy == "cold_both":
        return _make_cold_both_split(
            data=data,
            fold=fold,
            n_splits=n_splits,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    if strategy != "edge":
        raise ValueError(f"Unknown split strategy: {strategy}")

    hs_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    hs_trainval_idx, hs_test_idx = list(hs_kfold.split(data.hs_edges))[fold]
    hs_train_idx, hs_val_idx = train_test_split(
        hs_trainval_idx,
        test_size=validation_fraction,
        random_state=seed + fold,
        shuffle=True,
    )

    hdi_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    hdi_trainval_idx, hdi_test_idx = list(
        hdi_kfold.split(data.hdi_edges, data.hdi_labels)
    )[fold]
    hdi_train_idx, hdi_val_idx = train_test_split(
        hdi_trainval_idx,
        test_size=validation_fraction,
        random_state=seed + 1000 + fold,
        shuffle=True,
        stratify=data.hdi_labels[hdi_trainval_idx],
    )
    return FoldSplit(
        hs_train=np.asarray(hs_train_idx, dtype=np.int64),
        hs_val=np.asarray(hs_val_idx, dtype=np.int64),
        hs_test=np.asarray(hs_test_idx, dtype=np.int64),
        hdi_train=np.asarray(hdi_train_idx, dtype=np.int64),
        hdi_val=np.asarray(hdi_val_idx, dtype=np.int64),
        hdi_test=np.asarray(hdi_test_idx, dtype=np.int64),
    )


def _random_hs_indices(
    data: PreparedData,
    fold: int,
    n_splits: int,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trainval_idx, test_idx = list(kfold.split(data.hs_edges))[fold]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=validation_fraction,
        random_state=seed + fold,
        shuffle=True,
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def _entity_groups(
    entities: np.ndarray,
    fold: int,
    n_splits: int,
    seed: int,
    validation_fraction: float,
) -> tuple[set[int], set[int], set[int]]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.unique(entities))
    test_groups = set(map(int, np.array_split(shuffled, n_splits)[fold]))
    remaining = np.asarray(
        [entity for entity in shuffled if int(entity) not in test_groups],
        dtype=np.int64,
    )
    validation_count = max(1, int(round(len(remaining) * validation_fraction)))
    fold_rng = np.random.default_rng(seed + 1000 + fold)
    remaining = fold_rng.permutation(remaining)
    validation_groups = set(map(int, remaining[:validation_count]))
    train_groups = set(map(int, remaining[validation_count:]))
    return train_groups, validation_groups, test_groups


def _validate_hdi_partition(data: PreparedData, split: FoldSplit, name: str) -> None:
    for partition, idx in (
        ("train", split.hdi_train),
        ("validation", split.hdi_val),
        ("test", split.hdi_test),
    ):
        if len(idx) == 0:
            raise ValueError(f"{name} produced an empty HDI {partition} partition")
        if len(np.unique(data.hdi_labels[idx])) < 2:
            raise ValueError(
                f"{name} HDI {partition} partition has only one class"
            )


def _make_cold_drug_split(
    data: PreparedData,
    fold: int,
    n_splits: int,
    seed: int,
    validation_fraction: float,
) -> FoldSplit:
    hs_train, hs_val, hs_test = _random_hs_indices(
        data, fold, n_splits, seed, validation_fraction
    )
    train_drugs, validation_drugs, test_drugs = _entity_groups(
        data.hdi_edges[:, 1],
        fold,
        n_splits,
        seed + 7919,
        validation_fraction,
    )

    def indices(groups: set[int]) -> np.ndarray:
        return np.flatnonzero(
            np.isin(data.hdi_edges[:, 1], np.asarray(sorted(groups)))
        )

    split = FoldSplit(
        hs_train=hs_train,
        hs_val=hs_val,
        hs_test=hs_test,
        hdi_train=indices(train_drugs),
        hdi_val=indices(validation_drugs),
        hdi_test=indices(test_drugs),
    )
    _validate_hdi_partition(data, split, "Cold-drug split")
    return split


def _make_cold_both_split(
    data: PreparedData,
    fold: int,
    n_splits: int,
    seed: int,
    validation_fraction: float,
) -> FoldSplit:
    active_herbs = np.unique(
        np.concatenate([data.hs_edges[:, 0], data.hdi_edges[:, 0]])
    )
    train_herbs, validation_herbs, test_herbs = _entity_groups(
        active_herbs, fold, n_splits, seed, validation_fraction
    )
    train_drugs, validation_drugs, test_drugs = _entity_groups(
        data.hdi_edges[:, 1],
        fold,
        n_splits,
        seed + 7919,
        validation_fraction,
    )
    development_herbs = train_herbs | validation_herbs
    development_drugs = train_drugs | validation_drugs

    def hs_indices(groups: set[int]) -> np.ndarray:
        return np.flatnonzero(
            np.isin(data.hs_edges[:, 0], np.asarray(sorted(groups)))
        )

    def hdi_indices(herbs: set[int], drugs: set[int]) -> np.ndarray:
        herb_mask = np.isin(data.hdi_edges[:, 0], np.asarray(sorted(herbs)))
        drug_mask = np.isin(data.hdi_edges[:, 1], np.asarray(sorted(drugs)))
        return np.flatnonzero(herb_mask & drug_mask)

    hs_development = hs_indices(development_herbs)
    hs_train, hs_val = train_test_split(
        hs_development,
        test_size=validation_fraction,
        random_state=seed + fold,
        shuffle=True,
    )
    hdi_development = hdi_indices(development_herbs, development_drugs)
    hdi_train, hdi_val = train_test_split(
        hdi_development,
        test_size=validation_fraction,
        random_state=seed + 1000 + fold,
        shuffle=True,
        stratify=data.hdi_labels[hdi_development],
    )
    split = FoldSplit(
        hs_train=np.asarray(hs_train, dtype=np.int64),
        hs_val=np.asarray(hs_val, dtype=np.int64),
        hs_test=hs_indices(test_herbs),
        hdi_train=np.asarray(hdi_train, dtype=np.int64),
        hdi_val=np.asarray(hdi_val, dtype=np.int64),
        hdi_test=hdi_indices(test_herbs, test_drugs),
    )
    for name, idx in (
        ("hs_train", split.hs_train),
        ("hs_val", split.hs_val),
        ("hs_test", split.hs_test),
    ):
        if len(idx) == 0:
            raise ValueError(f"Cold-both split produced an empty partition: {name}")
    _validate_hdi_partition(data, split, "Cold-both split")
    return split


def _make_cold_herb_split(
    data: PreparedData,
    fold: int,
    n_splits: int,
    seed: int,
    validation_fraction: float,
) -> FoldSplit:
    active_herbs = np.unique(
        np.concatenate([data.hs_edges[:, 0], data.hdi_edges[:, 0]])
    )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(active_herbs)
    test_groups = set(map(int, np.array_split(shuffled, n_splits)[fold]))
    remaining = np.asarray(
        [herb for herb in shuffled if int(herb) not in test_groups], dtype=np.int64
    )
    validation_count = max(1, int(round(len(remaining) * validation_fraction)))
    fold_rng = np.random.default_rng(seed + 1000 + fold)
    remaining = fold_rng.permutation(remaining)
    validation_groups = set(map(int, remaining[:validation_count]))
    train_groups = set(map(int, remaining[validation_count:]))

    def indices(edges: np.ndarray, groups: set[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(edges[:, 0], np.asarray(sorted(groups))))

    split = FoldSplit(
        hs_train=indices(data.hs_edges, train_groups),
        hs_val=indices(data.hs_edges, validation_groups),
        hs_test=indices(data.hs_edges, test_groups),
        hdi_train=indices(data.hdi_edges, train_groups),
        hdi_val=indices(data.hdi_edges, validation_groups),
        hdi_test=indices(data.hdi_edges, test_groups),
    )
    for name, idx in (
        ("hs_train", split.hs_train),
        ("hs_val", split.hs_val),
        ("hs_test", split.hs_test),
        ("hdi_train", split.hdi_train),
        ("hdi_val", split.hdi_val),
        ("hdi_test", split.hdi_test),
    ):
        if len(idx) == 0:
            raise ValueError(f"Cold-herb split produced an empty partition: {name}")
    if len(np.unique(data.hdi_labels[split.hdi_val])) < 2:
        raise ValueError("Cold-herb validation partition has only one HDI class")
    if len(np.unique(data.hdi_labels[split.hdi_test])) < 2:
        raise ValueError("Cold-herb test partition has only one HDI class")
    return split


def known_symptoms_by_herb(hs_edges: np.ndarray) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for herb, symptom in hs_edges:
        result.setdefault(int(herb), set()).add(int(symptom))
    return result


def known_herbs_by_symptom(hs_edges: np.ndarray) -> dict[int, set[int]]:
    """Return the observed herb set for each symptom query."""
    result: dict[int, set[int]] = {}
    for herb, symptom in hs_edges:
        result.setdefault(int(symptom), set()).add(int(herb))
    return result


def sample_hs_negative_herbs(
    positive_edges: np.ndarray,
    known_positives: dict[int, set[int]],
    candidate_herbs: np.ndarray,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample unlabeled herbs while holding the symptom query fixed.

    Returns symptom indices, positive-herb indices, and negative-herb indices.
    ``known_positives`` controls the filtering scope and may be built from either
    all catalogued positives (filtered-PU protocol) or the current training fold.
    """
    symptoms: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    all_herbs = np.asarray(candidate_herbs, dtype=np.int64)
    candidate_cache: dict[int, np.ndarray] = {}
    for positive, symptom in positive_edges:
        symptom_i = int(symptom)
        if symptom_i not in candidate_cache:
            excluded = known_positives.get(symptom_i, set())
            candidate_cache[symptom_i] = np.asarray(
                [x for x in all_herbs if int(x) not in excluded], dtype=np.int64
            )
        candidates = candidate_cache[symptom_i]
        if len(candidates) == 0:
            continue
        sampled = rng.choice(
            candidates,
            size=negatives_per_positive,
            replace=len(candidates) < negatives_per_positive,
        )
        symptoms.extend([symptom_i] * negatives_per_positive)
        positives.extend([int(positive)] * negatives_per_positive)
        negatives.extend(sampled.tolist())
    return (
        np.asarray(symptoms, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
        np.asarray(negatives, dtype=np.int64),
    )


def sample_hs_negatives(
    positive_edges: np.ndarray,
    known_positives: dict[int, set[int]],
    n_symptoms: int,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    herbs: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    all_symptoms = np.arange(n_symptoms, dtype=np.int64)
    candidate_cache: dict[int, np.ndarray] = {}
    for herb, positive in positive_edges:
        herb_i = int(herb)
        if herb_i not in candidate_cache:
            excluded = known_positives.get(herb_i, set())
            candidate_cache[herb_i] = np.asarray(
                [x for x in all_symptoms if int(x) not in excluded], dtype=np.int64
            )
        candidates = candidate_cache[herb_i]
        if len(candidates) == 0:
            continue
        sampled = rng.choice(
            candidates,
            size=negatives_per_positive,
            replace=len(candidates) < negatives_per_positive,
        )
        herbs.extend([herb_i] * negatives_per_positive)
        positives.extend([int(positive)] * negatives_per_positive)
        negatives.extend(sampled.tolist())
    return (
        np.asarray(herbs, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
        np.asarray(negatives, dtype=np.int64),
    )


def split_audit(data: PreparedData, split: FoldSplit, strategy: str = "edge") -> dict:
    def edge_set(edges: np.ndarray, indices: np.ndarray) -> set[tuple[int, int]]:
        return {tuple(map(int, pair)) for pair in edges[indices]}

    hs_train = edge_set(data.hs_edges, split.hs_train)
    hs_val = edge_set(data.hs_edges, split.hs_val)
    hs_test = edge_set(data.hs_edges, split.hs_test)
    hdi_train = edge_set(data.hdi_edges, split.hdi_train)
    hdi_val = edge_set(data.hdi_edges, split.hdi_val)
    hdi_test = edge_set(data.hdi_edges, split.hdi_test)
    audit = {
        "strategy": strategy,
        "hs_train_edges": len(hs_train),
        "hs_val_edges": len(hs_val),
        "hs_test_edges": len(hs_test),
        "hs_train_val_overlap": len(hs_train & hs_val),
        "hs_train_test_overlap": len(hs_train & hs_test),
        "hs_val_test_overlap": len(hs_val & hs_test),
        "hdi_train_pairs": len(hdi_train),
        "hdi_val_pairs": len(hdi_val),
        "hdi_test_pairs": len(hdi_test),
        "hdi_train_val_overlap": len(hdi_train & hdi_val),
        "hdi_train_test_overlap": len(hdi_train & hdi_test),
        "hdi_val_test_overlap": len(hdi_val & hdi_test),
        "hdi_train_risk": int(data.hdi_labels[split.hdi_train].sum()),
        "hdi_val_risk": int(data.hdi_labels[split.hdi_val].sum()),
        "hdi_test_risk": int(data.hdi_labels[split.hdi_test].sum()),
    }
    if strategy in {"cold_herb", "cold_both"}:
        train_herbs = set(data.hs_edges[split.hs_train, 0]) | set(
            data.hdi_edges[split.hdi_train, 0]
        )
        val_herbs = set(data.hs_edges[split.hs_val, 0]) | set(
            data.hdi_edges[split.hdi_val, 0]
        )
        test_herbs = set(data.hs_edges[split.hs_test, 0]) | set(
            data.hdi_edges[split.hdi_test, 0]
        )
        audit.update(
            {
                "cold_herb_train_entities": len(train_herbs),
                "cold_herb_val_entities": len(val_herbs),
                "cold_herb_test_entities": len(test_herbs),
                "cold_herb_train_val_overlap": len(train_herbs & val_herbs),
                "cold_herb_train_test_overlap": len(train_herbs & test_herbs),
                "cold_herb_val_test_overlap": len(val_herbs & test_herbs),
            }
        )
    if strategy in {"cold_drug", "cold_both"}:
        train_drugs = set(data.hdi_edges[split.hdi_train, 1])
        val_drugs = set(data.hdi_edges[split.hdi_val, 1])
        test_drugs = set(data.hdi_edges[split.hdi_test, 1])
        audit.update(
            {
                "cold_drug_train_entities": len(train_drugs),
                "cold_drug_val_entities": len(val_drugs),
                "cold_drug_test_entities": len(test_drugs),
                "cold_drug_train_val_overlap": len(train_drugs & val_drugs),
                "cold_drug_train_test_overlap": len(train_drugs & test_drugs),
                "cold_drug_val_test_overlap": len(val_drugs & test_drugs),
            }
        )
    return audit
