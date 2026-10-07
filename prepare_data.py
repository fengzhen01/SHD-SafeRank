from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Cannot decode {path}")


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_herb_name(value) -> str:
    value = unicodedata.normalize("NFKC", clean_text(value))
    return "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))


def row_l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def hashed_relation_features(
    relations: pd.DataFrame,
    entity_column: str,
    item_column: str,
    n_entities: int,
    dimensions: int = 256,
) -> np.ndarray:
    """Create label-free fixed-size features from entity-item relations."""
    matrix = np.zeros((n_entities, dimensions), dtype=np.float32)
    pairs = relations[[entity_column, item_column]].drop_duplicates()
    for entity, item in pairs.itertuples(index=False):
        entity_i = int(entity)
        item_i = int(item)
        bucket = int((item_i * 2654435761) % dimensions)
        sign = 1.0 if ((item_i * 2246822519) & 1) == 0 else -1.0
        matrix[entity_i, bucket] += sign
    return row_l2_normalize(matrix)


def load_indexed_features(path: Path) -> tuple[list[str], np.ndarray]:
    frame = read_csv(path)
    ids = frame.iloc[:, 0].astype(str).tolist()
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return ids, row_l2_normalize(values.to_numpy(dtype=np.float32))


def stable_string_set_features(
    entity_items: dict[str, set[str]],
    entity_ids: list[str],
    dimensions: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(entity_ids), dimensions), dtype=np.float32)
    mask = np.zeros(len(entity_ids), dtype=np.float32)
    for entity_index, entity_id in enumerate(entity_ids):
        items = entity_items.get(str(entity_id), set())
        if not items:
            continue
        mask[entity_index] = 1.0
        for item in items:
            digest = hashlib.blake2b(
                str(item).encode("utf-8"), digest_size=8
            ).digest()
            bucket = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            matrix[entity_index, bucket] += sign
    return row_l2_normalize(matrix), mask


def load_positive_drugbank_targets(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.rstrip("\n").split()
            if len(fields) < 5 or fields[-1] != "1":
                continue
            drugbank_id, target_id = fields[0], fields[1]
            result.setdefault(drugbank_id, set()).add(target_id)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SHD-SafeRank data")
    parser.add_argument("--mhgnn-data", type=Path, required=True)
    parser.add_argument("--hdi-data", type=Path, required=True)
    parser.add_argument("--drugbank-dti", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def aggregate_hdi_evidence(
    dhi_pairs: pd.DataFrame, interactions: pd.DataFrame
) -> pd.DataFrame:
    base = dhi_pairs[
        ["Drug_ID", "Drug_Name", "Food_Herb_ID", "Food_Herb_Name"]
    ].drop_duplicates(["Drug_ID", "Food_Herb_ID"])
    evidence = interactions.merge(
        base[["Drug_ID", "Food_Herb_ID"]],
        on=["Drug_ID", "Food_Herb_ID"],
        how="inner",
    )
    evidence["Effect_clean"] = evidence["Effect"].map(clean_text)
    evidence["PMID_clean"] = evidence["PMID"].map(
        lambda value: "" if pd.isna(value) else str(int(float(value)))
    )

    rows: list[dict] = []
    for (drug_id, herb_id), group in evidence.groupby(
        ["Drug_ID", "Food_Herb_ID"], sort=False
    ):
        effects = sorted({effect for effect in group["Effect_clean"] if effect})
        effect_set = set(effects)
        has_harmful = "Harmful" in effect_set
        has_negative = "Negative" in effect_set
        has_risk = has_harmful or has_negative
        has_no_effect = "No Effect" in effect_set
        only_no_effect = effect_set == {"No Effect"}
        conflict = has_no_effect and len(effect_set) > 1
        if has_risk:
            supervision_label = 1
            semantic_class = "risk_evidence"
        elif only_no_effect:
            supervision_label = 0
            semantic_class = "no_effect_only"
        elif "Positive" in effect_set or "Possible" in effect_set:
            supervision_label = -1
            semantic_class = "unlabeled_positive_or_possible"
        else:
            supervision_label = -1
            semantic_class = "unlabeled_other"
        rows.append(
            {
                "Drug_ID": drug_id,
                "Food_Herb_ID": herb_id,
                "effect_categories": " | ".join(effects),
                "evidence_record_count": int(len(group)),
                "pmid_count": int(
                    group.loc[group["PMID_clean"] != "", "PMID_clean"].nunique()
                ),
                "has_harmful": int(has_harmful),
                "has_negative": int(has_negative),
                "has_risk": int(has_risk),
                "only_no_effect": int(only_no_effect),
                "conflicting_effects": int(conflict),
                "risk_label": supervision_label,
                "semantic_class": semantic_class,
                "sample_weight": 0.5 if has_risk and conflict else 1.0,
            }
        )
    aggregated = pd.DataFrame(rows)
    result = base.merge(
        aggregated, on=["Drug_ID", "Food_Herb_ID"], how="left"
    )
    result["risk_label"] = result["risk_label"].fillna(-1).astype(int)
    result["semantic_class"] = result["semantic_class"].fillna("unlabeled_missing")
    result["effect_categories"] = result["effect_categories"].fillna("")
    result["sample_weight"] = result["sample_weight"].fillna(1.0).astype(float)
    for column in [
        "evidence_record_count",
        "pmid_count",
        "has_harmful",
        "has_negative",
        "has_risk",
        "only_no_effect",
        "conflicting_effects",
    ]:
        result[column] = result[column].fillna(0).astype(int)
    return result


def main() -> None:
    args = parse_args()
    hit_dir = args.mhgnn_data / "HIT"
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    hit_herbs = pd.read_excel(hit_dir / "original_data" / "herb.xlsx")
    hit_names = read_csv(hit_dir / "original_data" / "herb_chinese.csv")
    hit_symptoms = pd.read_excel(hit_dir / "original_data" / "symptom.xlsx")
    hs_raw = pd.read_excel(hit_dir / "herb_symptom.xlsx")
    hs_edges = hs_raw.drop_duplicates(["node_id_herb", "node_id_symptom"]).copy()
    herb_target = pd.read_excel(hit_dir / "herb_target.xlsx")
    symptom_gene = pd.read_excel(hit_dir / "symptom_gene.xlsx")

    dhi_raw = read_csv(args.hdi_data / "DHI.csv")
    dhi_pairs = dhi_raw.drop_duplicates(["Drug_ID", "Food_Herb_ID"]).copy()
    herb_info = read_csv(args.hdi_data / "Herb Information.csv")
    drug_info = read_csv(args.hdi_data / "Drug Information.csv")
    interactions = read_csv(args.hdi_data / "Interaction Information.csv")
    hdi_labels = aggregate_hdi_evidence(dhi_pairs, interactions)

    hit = hit_herbs.merge(hit_names, on="tcm_id", how="left")
    hit["normalized_name"] = hit["herb_name"].map(normalize_herb_name)
    info = herb_info.copy()
    info["normalized_name"] = info["Herb_Chinese_Name"].map(normalize_herb_name)
    info = info[info["normalized_name"] != ""].drop_duplicates(
        ["normalized_name", "FHDI_Herb_ID"]
    )
    name_candidates = {
        name: sorted(group["FHDI_Herb_ID"].unique().tolist())
        for name, group in info.groupby("normalized_name")
    }
    info_by_id = (
        info.sort_values("FHDI_Herb_ID")
        .drop_duplicates("FHDI_Herb_ID")
        .set_index("FHDI_Herb_ID")
    )

    mapping_rows = []
    accepted_hit_to_hdi: dict[int, str] = {}
    accepted_hdi_to_hit: dict[str, int] = {}
    for row in hit.sort_values("node_id").itertuples(index=False):
        candidates = name_candidates.get(row.normalized_name, [])
        status = (
            "unambiguous_exact"
            if len(candidates) == 1
            else "ambiguous_exact"
            if len(candidates) > 1
            else "no_exact_match"
        )
        hdi_id = candidates[0] if status == "unambiguous_exact" else ""
        if hdi_id:
            accepted_hit_to_hdi[int(row.node_id)] = hdi_id
            accepted_hdi_to_hit[hdi_id] = int(row.node_id)
        details = info_by_id.loc[hdi_id] if hdi_id else None
        mapping_rows.append(
            {
                "hit_herb_node_id": int(row.node_id),
                "hit_tcm_id": clean_text(row.tcm_id),
                "hit_chinese_name": clean_text(row.herb_name),
                "normalized_name": row.normalized_name,
                "mapping_status": status,
                "candidate_count": len(candidates),
                "hdi_herb_id": hdi_id,
                "hdi_chinese_name": clean_text(details["Herb_Chinese_Name"])
                if details is not None
                else "",
                "hdi_english_name": clean_text(details["Herb_English_Name"])
                if details is not None
                else "",
            }
        )
    crosswalk = pd.DataFrame(mapping_rows)

    # A shared herb receives one unified index. Unmatched and ambiguous entities
    # remain dataset-specific to prevent false merges.
    unified_keys: list[str] = []
    hit_to_key: dict[int, str] = {}
    for row in hit.sort_values("node_id").itertuples(index=False):
        node_id = int(row.node_id)
        hdi_id = accepted_hit_to_hdi.get(node_id)
        key = (
            f"shared|{clean_text(row.tcm_id)}|{hdi_id}"
            if hdi_id
            else f"hit|{clean_text(row.tcm_id)}"
        )
        hit_to_key[node_id] = key
        unified_keys.append(key)

    hdi_to_key: dict[str, str] = {}
    for hdi_id in sorted(dhi_pairs["Food_Herb_ID"].unique()):
        hit_node = accepted_hdi_to_hit.get(hdi_id)
        if hit_node is not None:
            key = hit_to_key[hit_node]
        else:
            key = f"hdi|{hdi_id}"
            unified_keys.append(key)
        hdi_to_key[hdi_id] = key

    unified_keys = list(dict.fromkeys(unified_keys))
    unified_index = {key: idx for idx, key in enumerate(unified_keys)}
    symptom_ids = hit_symptoms.sort_values("node_id")["symptom"].astype(str).tolist()
    symptom_index = {
        int(row.node_id): idx
        for idx, row in enumerate(hit_symptoms.sort_values("node_id").itertuples())
    }
    drug_ids = sorted(dhi_pairs["Drug_ID"].unique().tolist())
    drug_index = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}

    hit_target_features = hashed_relation_features(
        herb_target,
        entity_column="herb",
        item_column="target",
        n_entities=len(hit_herbs),
    )
    symptom_features = hashed_relation_features(
        symptom_gene,
        entity_column="symptom",
        item_column="gene",
        n_entities=len(hit_symptoms),
    )
    hdi_herb_ids, hdi_herb_feature_values = load_indexed_features(
        args.hdi_data / "Herb_feature.csv"
    )
    all_drug_feature_ids, all_drug_feature_values = load_indexed_features(
        args.hdi_data / "Drug_feature.csv"
    )
    hdi_herb_feature_by_id = {
        herb_id: hdi_herb_feature_values[idx]
        for idx, herb_id in enumerate(hdi_herb_ids)
    }
    drug_feature_by_id = {
        drug_id: all_drug_feature_values[idx]
        for idx, drug_id in enumerate(all_drug_feature_ids)
    }
    herb_target_features = np.zeros((len(unified_keys), 256), dtype=np.float32)
    herb_component_features = np.zeros((len(unified_keys), 768), dtype=np.float32)
    herb_feature_mask = np.zeros((len(unified_keys), 2), dtype=np.float32)
    for hit_node, key in hit_to_key.items():
        unified_id = unified_index[key]
        herb_target_features[unified_id] = hit_target_features[int(hit_node)]
        herb_feature_mask[unified_id, 0] = 1.0
    for hdi_id, key in hdi_to_key.items():
        feature = hdi_herb_feature_by_id.get(str(hdi_id))
        if feature is not None:
            unified_id = unified_index[key]
            herb_component_features[unified_id] = feature
            herb_feature_mask[unified_id, 1] = 1.0
    drug_features = np.zeros((len(drug_ids), 768), dtype=np.float32)
    drug_feature_mask = np.zeros(len(drug_ids), dtype=np.float32)
    for drug_id, index in drug_index.items():
        feature = drug_feature_by_id.get(str(drug_id))
        if feature is not None:
            drug_features[index] = feature
            drug_feature_mask[index] = 1.0
    drug_target_features = np.zeros((len(drug_ids), 256), dtype=np.float32)
    drug_target_feature_mask = np.zeros(len(drug_ids), dtype=np.float32)
    if args.drugbank_dti is not None:
        positive_targets = load_positive_drugbank_targets(args.drugbank_dti)
        fhdi_to_drugbank = (
            drug_info[["FHDI_Drug_ID", "DrugBank_ID"]]
            .dropna(subset=["DrugBank_ID"])
            .drop_duplicates("FHDI_Drug_ID")
            .set_index("FHDI_Drug_ID")["DrugBank_ID"]
            .astype(str)
            .to_dict()
        )
        drug_target_sets = {
            str(drug_id): positive_targets.get(
                fhdi_to_drugbank.get(str(drug_id), ""), set()
            )
            for drug_id in drug_ids
        }
        drug_target_features, drug_target_feature_mask = stable_string_set_features(
            drug_target_sets, [str(drug_id) for drug_id in drug_ids]
        )

    hs_array = np.asarray(
        [
            [
                unified_index[hit_to_key[int(row.node_id_herb)]],
                symptom_index[int(row.node_id_symptom)],
            ]
            for row in hs_edges.itertuples(index=False)
        ],
        dtype=np.int64,
    )

    supervised = hdi_labels[hdi_labels["risk_label"] >= 0].copy()
    hdi_edges = np.asarray(
        [
            [
                unified_index[hdi_to_key[row.Food_Herb_ID]],
                drug_index[row.Drug_ID],
            ]
            for row in supervised.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    hdi_y = supervised["risk_label"].to_numpy(dtype=np.int64)
    hdi_weight = supervised["sample_weight"].to_numpy(dtype=np.float32)

    unlabeled = hdi_labels[hdi_labels["risk_label"] < 0].copy()
    unlabeled_edges = np.asarray(
        [
            [
                unified_index[hdi_to_key[row.Food_Herb_ID]],
                drug_index[row.Drug_ID],
            ]
            for row in unlabeled.itertuples(index=False)
        ],
        dtype=np.int64,
    )

    np.savez_compressed(
        output_dir / "data.npz",
        hs_edges=hs_array,
        hdi_edges=hdi_edges,
        hdi_labels=hdi_y,
        hdi_weights=hdi_weight,
        hdi_unlabeled_edges=unlabeled_edges,
        unified_herb_keys=np.asarray(unified_keys, dtype="U80"),
        symptom_ids=np.asarray(symptom_ids, dtype="U40"),
        drug_ids=np.asarray(drug_ids, dtype="U40"),
        herb_target_features=herb_target_features,
        herb_component_features=herb_component_features,
        herb_feature_mask=herb_feature_mask,
        symptom_features=symptom_features,
        drug_features=drug_features,
        drug_feature_mask=drug_feature_mask,
        drug_target_features=drug_target_features,
        drug_target_feature_mask=drug_target_feature_mask,
    )

    crosswalk["unified_herb_key"] = crosswalk.apply(
        lambda row: hit_to_key[int(row["hit_herb_node_id"])], axis=1
    )
    crosswalk["in_hdi_relation_table"] = crosswalk["hdi_herb_id"].isin(
        set(dhi_pairs["Food_Herb_ID"])
    ).astype(int)
    active_hit_nodes = set(hs_edges["node_id_herb"].astype(int))
    crosswalk["in_hit_relation_table"] = crosswalk["hit_herb_node_id"].isin(
        active_hit_nodes
    ).astype(int)
    crosswalk["usable_shared_herb"] = (
        (crosswalk["mapping_status"] == "unambiguous_exact")
        & (crosswalk["in_hdi_relation_table"] == 1)
        & (crosswalk["in_hit_relation_table"] == 1)
    ).astype(int)
    crosswalk.to_csv(output_dir / "herb_crosswalk.csv", index=False, encoding="utf-8-sig")
    hdi_labels.to_csv(output_dir / "hdi_pair_labels.csv", index=False, encoding="utf-8-sig")

    usable = crosswalk[crosswalk["usable_shared_herb"] == 1][
        [
            "hit_herb_node_id",
            "hit_tcm_id",
            "hit_chinese_name",
            "hdi_herb_id",
            "unified_herb_key",
        ]
    ]
    symptoms = hit_symptoms.rename(
        columns={"node_id": "node_id_symptom", "symptom": "symptom_umls_id"}
    )
    left = hs_edges.merge(symptoms, on="node_id_symptom").merge(
        usable,
        left_on="node_id_herb",
        right_on="hit_herb_node_id",
        how="inner",
    )
    weak_paths = left.merge(
        hdi_labels,
        left_on="hdi_herb_id",
        right_on="Food_Herb_ID",
        how="inner",
    )
    weak_paths["path_semantics"] = "composed_binary_path_not_validated_ternary_fact"
    weak_paths.to_csv(output_dir / "weak_paths.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "dataset": "SHD-SafeRank v0.1",
        "hit_raw_edges": int(len(hs_raw)),
        "hit_unique_edges": int(len(hs_edges)),
        "hdi_raw_rows": int(len(dhi_raw)),
        "hdi_unique_pairs": int(len(dhi_pairs)),
        "hdi_risk_positive_pairs": int((hdi_labels["risk_label"] == 1).sum()),
        "hdi_no_effect_only_pairs": int((hdi_labels["risk_label"] == 0).sum()),
        "hdi_unlabeled_pairs": int((hdi_labels["risk_label"] < 0).sum()),
        "unified_herbs": int(len(unified_keys)),
        "symptoms": int(len(symptom_ids)),
        "drugs": int(len(drug_ids)),
        "unambiguous_exact_mappings": int(
            (crosswalk["mapping_status"] == "unambiguous_exact").sum()
        ),
        "usable_shared_herbs": int(crosswalk["usable_shared_herb"].sum()),
        "weak_paths": int(len(weak_paths)),
        "feature_provenance": {
            "herb_target_features": "256-dimensional fixed hash projection of HIT herb-target relations",
            "herb_component_features": "row-normalized 768-dimensional HDI Herb_feature.csv",
            "symptom_features": "256-dimensional fixed hash projection of HIT symptom-gene relations",
            "drug_features": "row-normalized 768-dimensional HDI Drug_feature.csv",
            "drug_target_features": (
                "256-dimensional fixed hash projection of positive DrugBank DTI relations"
                if args.drugbank_dti is not None
                else "not provided"
            ),
            "excluded": "MHGNN pre-generated bipartite herb/symptom embeddings are excluded because they may encode target herb-symptom edges",
        },
        "feature_coverage": {
            "herbs_with_target_features": int(herb_feature_mask[:, 0].sum()),
            "herbs_with_component_features": int(herb_feature_mask[:, 1].sum()),
            "herbs_with_both_feature_types": int(
                np.logical_and(
                    herb_feature_mask[:, 0] > 0,
                    herb_feature_mask[:, 1] > 0,
                ).sum()
            ),
            "symptoms_with_gene_features": int(
                (np.linalg.norm(symptom_features, axis=1) > 0).sum()
            ),
            "drugs_with_molecular_features": int(drug_feature_mask.sum()),
            "drugs_with_target_features": int(drug_target_feature_mask.sum()),
        },
        "risk_label_policy": {
            "positive": "at least one Harmful or Negative evidence record",
            "negative": "all available evidence records are No Effect",
            "unlabeled": "all remaining pairs",
            "conflicting_risk_weight": 0.5,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
