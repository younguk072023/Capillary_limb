"""Restyle completed analysis figures from saved data, without refitting models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dinov3_full_clustering import plot_scores
from dinov3_cluster_projection import _save_projection_plot, save_umap_tsne_comparison
from dinov3_cluster_interpretation import render_interpretation_figures
from dinov3_cluster_plot_style import CLUSTER_COLORS, plot_style_metadata


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def redraw(analysis: Path) -> dict:
    analysis = analysis.resolve()
    metadata_path = analysis / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = pd.read_csv(analysis / "clustered_valid_images.csv", encoding="utf-8-sig")
    coordinates = pd.read_csv(analysis / "projection_coordinates.csv", encoding="utf-8-sig")
    scores = pd.read_csv(analysis / "cluster_k_scores.csv", encoding="utf-8-sig")
    if selected["Filename"].tolist() != coordinates["Filename"].tolist():
        raise ValueError("Projection rows do not match saved assignments")
    if not np.isfinite(coordinates[["PC1", "PC2", "UMAP1", "UMAP2"]].to_numpy()).all():
        raise ValueError("Nonfinite saved projection coordinates")
    interpretation_dir = analysis / "interpretation"
    with np.load(interpretation_dir / "shap_values.npz", allow_pickle=False) as saved:
        shap_values = saved["values"].copy()
        classes = saved["class_labels"].copy()
        labels = saved["kmeans_labels"].copy()
        features = saved["feature_names"].tolist()
        np.testing.assert_array_equal(saved["filenames"], selected["Filename"].to_numpy())
        np.testing.assert_array_equal(labels, selected["kmeans_cluster"].to_numpy())
        x = pd.DataFrame(saved["raw_feature_values"].copy(), columns=features)
        np.testing.assert_allclose(x.to_numpy(), selected[features].to_numpy(), rtol=1e-12, atol=1e-12)
    confusion = pd.read_csv(interpretation_dir / "rf_training_confusion_matrix.csv", index_col=0)
    class_names = [f"cluster_{label}" for label in classes]
    if confusion.index.tolist() != class_names or confusion.columns.tolist() != class_names:
        raise ValueError("Confusion matrix class order disagrees with saved SHAP")
    matrix = confusion.to_numpy(dtype=int)
    if matrix.sum() != len(selected):
        raise ValueError("Confusion matrix count disagrees with included rows")

    # These scientific results must remain byte-for-byte identical during redraw.
    protected = {str(path.relative_to(analysis)): digest(path)
                 for pattern in ("*.csv", "*.npz", "*.joblib") for path in analysis.rglob(pattern)}
    print("Redrawing saved PCA/UMAP and K-score figures...", flush=True)
    plot_scores(scores, analysis, int(metadata["k"]))
    pca = metadata["projection"]["pca"]
    pca_axes = (f"PC1 ({pca['pc1'] * 100:.1f}%)", f"PC2 ({pca['pc2'] * 100:.1f}%)")
    for method in ("kmeans", "ward"):
        method_labels = selected[f"{method}_cluster"].to_numpy(dtype=int)
        np.testing.assert_array_equal(method_labels, coordinates[f"{method}_cluster"].to_numpy())
        _save_projection_plot(coordinates[["PC1", "PC2"]].to_numpy(), method_labels,
                              method, "pca", analysis, pca_axes)
        _save_projection_plot(coordinates[["UMAP1", "UMAP2"]].to_numpy(), method_labels,
                              method, "umap", analysis, ("UMAP 1", "UMAP 2"))
    tsne_path = analysis / "tsne_coordinates.csv"
    if tsne_path.is_file():
        tsne = pd.read_csv(tsne_path, encoding="utf-8-sig", float_precision="round_trip")
        np.testing.assert_array_equal(tsne["Filename"], selected["Filename"])
        tsne_xy = tsne[["TSNE1", "TSNE2"]].to_numpy()
        if not np.isfinite(tsne_xy).all():
            raise ValueError("Nonfinite saved t-SNE coordinates")
        for method in ("kmeans", "ward"):
            labels_tsne = selected[f"{method}_cluster"].to_numpy(dtype=int)
            np.testing.assert_array_equal(tsne[f"{method}_cluster"], labels_tsne)
            _save_projection_plot(tsne_xy, labels_tsne, method, "tsne", analysis, ("t-SNE 1", "t-SNE 2"))
        save_umap_tsne_comparison(coordinates[["UMAP1", "UMAP2"]].to_numpy(), tsne_xy,
                                  selected["kmeans_cluster"].to_numpy(dtype=int),
                                  analysis / "kmeans_umap_tsne_comparison")
    print("Redrawing saved feature distributions, RF confusion and SHAP...", flush=True)
    render_interpretation_figures(
        x, labels, classes, shap_values, matrix, float(np.trace(matrix) / matrix.sum()),
        interpretation_dir, seed=int(metadata["seed"]),
    )
    for relative_path, before in protected.items():
        if digest(analysis / relative_path) != before:
            raise RuntimeError(f"Numeric artifact changed during redraw: {relative_path}")

    style = plot_style_metadata()
    code_files = ("dinov3_full_clustering.py", "dinov3_cluster_projection.py",
                  "dinov3_cluster_interpretation.py", "dinov3_cluster_plot_style.py",
                  "redraw_dinov3_cluster_figures.py")
    timestamp = datetime.now(timezone.utc).isoformat()
    figure_files = sorted(str(p.relative_to(analysis)) for extension in ("png", "pdf", "tiff")
                          for p in analysis.rglob(f"*.{extension}"))
    render_metadata = {
        "rendered_at_utc": timestamp,
        "analysis_rerun": False,
        "projection_coordinates_reused": True,
        "shap_values_reused": True,
        "numeric_artifacts_unchanged": True,
        "numeric_artifact_sha256": protected,
        "plot_format": style,
        "rendering_code_sha256": {name: digest(Path(__file__).with_name(name)) for name in code_files},
        "figures": figure_files,
    }
    (analysis / "figure_style_metadata.json").write_text(
        json.dumps(render_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    updated_tsne = None
    tsne_metadata_path = analysis / "tsne_metadata.json"
    if tsne_metadata_path.is_file():
        updated_tsne = json.loads(tsne_metadata_path.read_text(encoding="utf-8"))
        updated_tsne["previous_plot_format"] = updated_tsne.get("plot_format")
        updated_tsne["plot_format"] = style
        updated_tsne["figure_updated_at_utc"] = timestamp
        updated_tsne["figure_style_metadata"] = "figure_style_metadata.json"
        tsne_metadata_path.write_text(json.dumps(updated_tsne, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, nested_key in (("projection_metadata.json", "projection"),
                                 ("interpretation/interpretation_metadata.json", "interpretation")):
        path = analysis / filename
        section = json.loads(path.read_text(encoding="utf-8"))
        section["previous_plot_format"] = section.get("plot_format")
        section["plot_format"] = style
        section["figure_style_metadata"] = "figure_style_metadata.json"
        section["figure_updated_at_utc"] = timestamp
        if nested_key == "projection":
            if updated_tsne is not None:
                section["tsne"] = updated_tsne
            section["cluster_colors"] = {str(i): CLUSTER_COLORS[i] for i in
                                         sorted(set(selected["kmeans_cluster"]) | set(selected["ward_cluster"]))}
            section["files"]["figures"] = [name for name in figure_files
                                           if name.startswith(("kmeans_", "ward_"))]
        path.write_text(json.dumps(section, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata[nested_key] = section
    # Keep the historical analysis code hashes; record plot-only code separately.
    metadata["plot_format"] = style
    metadata["figure_style_metadata"] = "figure_style_metadata.json"
    metadata["figure_updated_at_utc"] = timestamp
    metadata["rendering_code_sha256"] = render_metadata["rendering_code_sha256"]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(figure_files), "numeric_artifacts_unchanged": len(protected),
                      "font": style["font"], "titles": style["titles"]}, indent=2), flush=True)
    return render_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Existing completed analysis; overwrite figures only and update style metadata")
    args = parser.parse_args()
    redraw(args.analysis_dir)
