# Data provenance and usage notes

## Included processed files

The repository includes a compact processed release in `data/processed/` (approximately 3.3 MB):

- `data.npz`: numerical features, binary relations, evidence weights, and entity indices;
- `metadata.json`: dataset counts, feature provenance, and risk-label policy;
- `herb_crosswalk.csv`: conservative exact-name mappings;
- `hdi_pair_labels.csv`: pair-level aggregated evidence semantics;
- `weak_paths.csv`: exploratory symptom–herb–drug paths composed from two binary relations.

No file in this directory should be interpreted as a clinical recommendation dataset.

## Source components

The processed release was constructed from publicly available research resources used by the project:

1. HIT herb–symptom associations and herb/symptom target information distributed with the MHGNN research data;
2. public herb–drug interaction evidence used by the HDI project;
3. independent DrugBank-derived drug–target relationships used only as external entity features.

Users remain responsible for complying with the terms of the original data providers. DrugBank-derived information may be subject to DrugBank licensing conditions and should not be treated as unrestricted raw DrugBank redistribution.

## Label policy

- **Risk positive:** at least one `Harmful` or `Negative` evidence record.
- **Non-risk reference:** all evidence records for the pair are `No Effect`.
- **Unlabeled:** `Possible`, `Positive`, or other evidence without a definite risk label.
- **Conflicting pair:** a risk-priority positive with reduced evidence weight when risk and `No Effect` evidence coexist.

Unlabeled pairs are excluded from the supervised HDI loss.

## Scientific boundary

The 4,979 symptom–herb–drug paths in `weak_paths.csv` are joins of observed symptom–herb and herb–drug relations. They are not validated ternary clinical facts and are not used as direct ternary supervision.
