from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from paper_plotting import configure_matplotlib, save_figure_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biological and network-level analyses")
    parser.add_argument("--mhgnn-data", type=Path, required=True)
    parser.add_argument("--processed-data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def set_map(frame: pd.DataFrame, key: str, item: str) -> dict[int, set[int]]:
    return {
        int(entity): set(group[item].astype(int).tolist())
        for entity, group in frame.groupby(key)
    }


def overlap_rows(
    pairs: np.ndarray,
    herb_targets: dict[int, set[int]],
    symptom_genes: dict[int, set[int]],
    group: str,
) -> list[dict]:
    rows: list[dict] = []
    for herb, symptom in pairs:
        left = herb_targets.get(int(herb), set())
        right = symptom_genes.get(int(symptom), set())
        intersection = len(left & right)
        union = len(left | right)
        cosine = intersection / np.sqrt(max(len(left) * len(right), 1))
        rows.append(
            {
                "group": group,
                "herb_index": int(herb),
                "symptom_index": int(symptom),
                "shared_targets": intersection,
                "jaccard": intersection / union if union else 0.0,
                "cosine": cosine,
                "herb_target_count": len(left),
                "symptom_gene_count": len(right),
            }
        )
    return rows


def bootstrap_difference(
    positive: np.ndarray, random: np.ndarray, repeats: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    observed = float(positive.mean() - random.mean())
    differences = np.empty(repeats, dtype=float)
    for index in range(repeats):
        p = rng.choice(positive, size=len(positive), replace=True)
        r = rng.choice(random, size=len(random), replace=True)
        differences[index] = p.mean() - r.mean()
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return observed, float(lower), float(upper)


def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values[values > 0]
    x = np.unique(values)
    return x, np.asarray([(values >= threshold).mean() for threshold in x])


def plot_topology(
    hs: pd.DataFrame,
    hdi: pd.DataFrame,
    metadata: dict,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    herb_degree = hs.groupby("node_id_herb").size().reindex(range(798), fill_value=0).to_numpy()
    symptom_degree = hs.groupby("node_id_symptom").size().reindex(range(174), fill_value=0).to_numpy()
    for values, label, color in (
        (herb_degree, "Herbs", "#2F855A"),
        (symptom_degree, "Symptoms", "#4C78A8"),
    ):
        x, y = ccdf(values)
        axes[0, 0].plot(x, y, marker="o", markersize=3, label=label, color=color)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Association degree")
    axes[0, 0].set_ylabel("P(Degree >= x)")
    axes[0, 0].set_title("Herb-symptom degree distribution")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)

    labelled = hdi[hdi["risk_label"].isin([0, 1])]
    herb_hdi_degree = labelled.groupby("Food_Herb_ID").size().to_numpy()
    drug_hdi_degree = labelled.groupby("Drug_ID").size().to_numpy()
    for values, label, color in (
        (herb_hdi_degree, "Herbs", "#2F855A"),
        (drug_hdi_degree, "Drugs", "#F58518"),
    ):
        x, y = ccdf(values)
        axes[0, 1].plot(x, y, marker="o", markersize=3, label=label, color=color)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Labelled interaction degree")
    axes[0, 1].set_ylabel("P(Degree >= x)")
    axes[0, 1].set_title("Herb-drug evidence degree distribution")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.2)

    effect_counts = {
        "Risk\nevidence": int((hdi["risk_label"] == 1).sum()),
        "No-effect\nonly": int((hdi["risk_label"] == 0).sum()),
        "Unlabelled/\nother": int((hdi["risk_label"] == -1).sum()),
    }
    axes[1, 0].bar(effect_counts.keys(), effect_counts.values(), color=["#C53030", "#2F855A", "#A0AEC0"])
    axes[1, 0].set_ylabel("Unique herb-drug pairs")
    axes[1, 0].set_title("Pair-level evidence semantics")
    axes[1, 0].grid(axis="y", alpha=0.2)
    for position, value in enumerate(effect_counts.values()):
        axes[1, 0].text(position, value, str(value), ha="center", va="bottom", fontsize=9)

    coverage = metadata["feature_coverage"]
    labels = ["Herb\ntarget", "Herb\ncomponent", "Symptom\ngene", "Drug\nmolecular", "Drug\ntarget"]
    values = [
        coverage["herbs_with_target_features"] / metadata["unified_herbs"],
        coverage["herbs_with_component_features"] / metadata["unified_herbs"],
        coverage["symptoms_with_gene_features"] / metadata["symptoms"],
        coverage["drugs_with_molecular_features"] / metadata["drugs"],
        coverage["drugs_with_target_features"] / metadata["drugs"],
    ]
    axes[1, 1].bar(labels, values, color=["#2F855A", "#72B7B2", "#4C78A8", "#F58518", "#E45756"])
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_ylabel("Coverage fraction")
    axes[1, 1].set_title("Entity-feature coverage")
    axes[1, 1].grid(axis="y", alpha=0.2)
    for position, value in enumerate(values):
        axes[1, 1].text(position, value, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    save_figure_bundle(fig, output)
    plt.close(fig)


def plot_overlap(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    metrics = [
        ("shared_targets", "Shared targets", False),
        ("jaccard", "Target Jaccard", True),
        ("cosine", "Target cosine", True),
    ]
    groups = ["Known association", "Random unlabelled"]
    colors = ["#4C78A8", "#A0AEC0"]
    for axis, (metric, label, log_y) in zip(axes, metrics):
        data = [frame.loc[frame["group"] == group, metric] for group in groups]
        parts = axis.violinplot(data, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.75)
        axis.set_xticks([1, 2], ["Known", "Random"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        row = summary[summary["metric"] == metric].iloc[0]
        axis.set_title(f"{label}\nmean diff={row['mean_difference']:.4f}")
        if log_y:
            upper = float(np.quantile(frame[metric], 0.995))
            axis.set_ylim(0.0, max(upper * 1.08, 1e-3))
    save_figure_bundle(fig, output)
    plt.close(fig)


def build_network(
    hs: pd.DataFrame,
    herbs: pd.DataFrame,
    symptoms: pd.DataFrame,
    crosswalk: pd.DataFrame,
    hdi: pd.DataFrame,
) -> tuple[nx.Graph, dict[str, tuple[float, float]], pd.DataFrame]:
    usable = crosswalk[crosswalk["usable_shared_herb"] == 1].copy()
    usable_nodes = set(usable["hit_herb_node_id"].astype(int))
    hs_sub = hs[hs["node_id_herb"].isin(usable_nodes)]
    hdi_labelled = hdi[hdi["risk_label"].isin([0, 1])]
    hit_to_hdi = usable.set_index("hit_herb_node_id")["hdi_herb_id"].astype(str).to_dict()
    hs_degree = hs_sub.groupby("node_id_herb").size()
    hdi_degree_by_id = hdi_labelled.groupby("Food_Herb_ID").size()
    score_rows = []
    for node in usable_nodes:
        hdi_id = hit_to_hdi.get(node, "")
        score_rows.append((node, int(hs_degree.get(node, 0)), int(hdi_degree_by_id.get(hdi_id, 0))))
    score_rows.sort(key=lambda row: row[1] + row[2], reverse=True)
    # A moderately expanded main-text network: enough entities to reveal hubs
    # and repeated neighbours while remaining legible on a journal page.
    selected_herbs = [row[0] for row in score_rows[:10]]

    herb_names = usable.set_index("hit_herb_node_id")["hit_chinese_name"].astype(str).to_dict()
    symptom_ids = symptoms.set_index("node_id")["symptom"].astype(str).to_dict()
    graph = nx.Graph()
    edge_rows = []
    selected_symptoms: set[int] = set()
    selected_drugs: set[str] = set()
    for herb in selected_herbs:
        herb_node = f"H:{herb}"
        graph.add_node(herb_node, kind="herb", label=herb_names.get(herb, str(herb)))
        symptom_neighbors = (
            hs_sub[hs_sub["node_id_herb"] == herb]["node_id_symptom"]
            .astype(int)
            .map(hs_sub.groupby("node_id_symptom").size())
            .sort_values(ascending=False)
            .index
        )
        symptom_neighbors = (
            hs_sub.loc[symptom_neighbors, "node_id_symptom"].astype(int).tolist()[:4]
        )
        for symptom in symptom_neighbors:
            selected_symptoms.add(symptom)
            symptom_node = f"S:{symptom}"
            graph.add_node(symptom_node, kind="symptom", label=symptom_ids.get(symptom, str(symptom)))
            graph.add_edge(herb_node, symptom_node, relation="efficacy", color="#4C78A8")
            edge_rows.append({"source": herb_node, "target": symptom_node, "relation": "herb-symptom association"})
        hdi_id = hit_to_hdi.get(herb, "")
        candidates = hdi_labelled[hdi_labelled["Food_Herb_ID"] == hdi_id].copy()
        risk_candidates = candidates[candidates["risk_label"] == 1].nlargest(2, "evidence_record_count")
        no_effect_candidates = candidates[candidates["risk_label"] == 0].nlargest(1, "evidence_record_count")
        candidates = pd.concat([risk_candidates, no_effect_candidates]).drop_duplicates(["Drug_ID"])
        for row in candidates.itertuples():
            drug_id = str(row.Drug_ID)
            selected_drugs.add(drug_id)
            drug_node = f"D:{drug_id}"
            label = str(row.Drug_Name)
            graph.add_node(drug_node, kind="drug", label=label)
            relation = "risk" if int(row.risk_label) == 1 else "no_effect"
            color = "#C53030" if relation == "risk" else "#2F855A"
            graph.add_edge(herb_node, drug_node, relation=relation, color=color)
            edge_rows.append({"source": herb_node, "target": drug_node, "relation": relation})

    pos: dict[str, tuple[float, float]] = {}
    symptom_nodes = [node for node, value in graph.nodes(data=True) if value["kind"] == "symptom"]
    herb_nodes = [node for node, value in graph.nodes(data=True) if value["kind"] == "herb"]
    drug_nodes = [node for node, value in graph.nodes(data=True) if value["kind"] == "drug"]
    for nodes, x in ((symptom_nodes, 0.0), (herb_nodes, 1.0), (drug_nodes, 2.0)):
        for y, node in zip(np.linspace(1, -1, max(len(nodes), 1)), nodes):
            pos[node] = (x, float(y))
    return graph, pos, pd.DataFrame(edge_rows)


def plot_network(graph: nx.Graph, pos: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axis = plt.subplots(figsize=(16, 10), constrained_layout=True)
    kind_style = {
        "symptom": ("#4C78A8", "s", 540),
        "herb": ("#F2C14E", "o", 760),
        "drug": ("#F58518", "D", 540),
    }
    for kind, (color, shape, size) in kind_style.items():
        nodes = [node for node, value in graph.nodes(data=True) if value["kind"] == kind]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=color, node_shape=shape, node_size=size, edgecolors="white", linewidths=0.8, ax=axis)
    edge_colors = [value["color"] for _, _, value in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=1.5, alpha=0.65, ax=axis)
    labels = {node: value["label"] for node, value in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6.5, ax=axis)
    axis.text(0, 1.12, "Symptoms", ha="center", fontsize=12, fontweight="bold")
    axis.text(1, 1.12, "Shared herbs", ha="center", fontsize=12, fontweight="bold")
    axis.text(2, 1.12, "Drugs", ha="center", fontsize=12, fontweight="bold")
    legend = [
        Line2D([0], [0], color="#4C78A8", lw=2, label="Known herb-symptom edge"),
        Line2D([0], [0], color="#C53030", lw=2, label="Harmful/Negative herb-drug evidence"),
        Line2D([0], [0], color="#2F855A", lw=2, label="No-effect-only herb-drug evidence"),
    ]
    axis.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8)
    axis.set_title("Exploratory network assembled from two audited binary relations", fontsize=13)
    axis.axis("off")
    save_figure_bundle(fig, output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    root = Path(__file__).resolve().parent
    mhgnn = args.mhgnn_data
    processed = args.processed_data if args.processed_data.is_absolute() else root / args.processed_data
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    hit = mhgnn / "HIT"
    hs = pd.read_excel(hit / "herb_symptom.xlsx").drop_duplicates(["node_id_herb", "node_id_symptom"])
    herb_target = pd.read_excel(hit / "herb_target.xlsx").drop_duplicates()
    symptom_gene = pd.read_excel(hit / "symptom_gene.xlsx").drop_duplicates()
    herbs = pd.read_excel(hit / "original_data" / "herb.xlsx")
    symptoms = pd.read_excel(hit / "original_data" / "symptom.xlsx")
    hdi = pd.read_csv(processed / "hdi_pair_labels.csv")
    crosswalk = pd.read_csv(processed / "herb_crosswalk.csv")
    metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))

    herb_targets = set_map(herb_target, "herb", "target")
    symptom_genes = set_map(symptom_gene, "symptom", "gene")
    known_pairs = hs[["node_id_herb", "node_id_symptom"]].to_numpy(dtype=int)
    known_set = set(map(tuple, known_pairs.tolist()))
    rng = np.random.default_rng(args.seed)
    random_pairs: list[tuple[int, int]] = []
    while len(random_pairs) < len(known_pairs):
        candidates = zip(
            rng.integers(0, len(herbs), size=len(known_pairs)),
            rng.integers(0, len(symptoms), size=len(known_pairs)),
        )
        for pair in candidates:
            pair = (int(pair[0]), int(pair[1]))
            if pair not in known_set and pair not in random_pairs:
                random_pairs.append(pair)
                if len(random_pairs) == len(known_pairs):
                    break
    overlap = pd.DataFrame(
        overlap_rows(known_pairs, herb_targets, symptom_genes, "Known association")
        + overlap_rows(np.asarray(random_pairs), herb_targets, symptom_genes, "Random unlabelled")
    )
    overlap.to_csv(output / "target_overlap_pairs.csv", index=False)
    summaries = []
    for metric in ["shared_targets", "jaccard", "cosine"]:
        positive = overlap.loc[overlap["group"] == "Known association", metric].to_numpy(float)
        random_values = overlap.loc[overlap["group"] == "Random unlabelled", metric].to_numpy(float)
        difference, lower, upper = bootstrap_difference(positive, random_values, args.bootstrap, rng)
        statistic, pvalue = mannwhitneyu(positive, random_values, alternative="two-sided")
        summaries.append(
            {
                "metric": metric,
                "known_mean": positive.mean(),
                "random_mean": random_values.mean(),
                "mean_difference": difference,
                "bootstrap_95ci_low": lower,
                "bootstrap_95ci_high": upper,
                "mannwhitney_u": statistic,
                "p_value": pvalue,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "target_overlap_summary.csv", index=False)

    topology_rows = []
    for entity, values in (
        ("HIT herbs", hs.groupby("node_id_herb").size().reindex(range(len(herbs)), fill_value=0)),
        ("HIT symptoms", hs.groupby("node_id_symptom").size().reindex(range(len(symptoms)), fill_value=0)),
        ("HDI labelled herbs", hdi[hdi["risk_label"].isin([0, 1])].groupby("Food_Herb_ID").size()),
        ("HDI labelled drugs", hdi[hdi["risk_label"].isin([0, 1])].groupby("Drug_ID").size()),
    ):
        topology_rows.append({"entity": entity, "count": len(values), "zero_degree": int((values == 0).sum()), "mean_degree": float(values.mean()), "median_degree": float(values.median()), "max_degree": int(values.max())})
    pd.DataFrame(topology_rows).to_csv(output / "network_topology_summary.csv", index=False)
    plot_topology(hs, hdi, metadata, output / "bioinformatics_topology.png")
    plot_overlap(overlap, summary, output / "target_overlap_analysis.png")
    graph, pos, edges = build_network(hs, herbs, symptoms, crosswalk, hdi)
    edges.to_csv(output / "exploratory_network_edges.csv", index=False)
    plot_network(graph, pos, output / "exploratory_tripartite_network.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
