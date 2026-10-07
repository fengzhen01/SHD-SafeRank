from __future__ import annotations

from pathlib import Path


def configure_matplotlib() -> None:
    """Use publication-friendly fonts while keeping SVG text editable."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_bundle(fig, output: Path, dpi: int = 300) -> list[Path]:
    """Save one Matplotlib figure as PNG, editable SVG, and vector PDF."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg"), stem.with_suffix(".pdf")]
    for path in paths:
        options = {"bbox_inches": "tight"}
        if path.suffix == ".png":
            options["dpi"] = dpi
        fig.savefig(path, **options)
    return paths
