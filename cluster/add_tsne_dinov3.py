"""Add t-SNE to a completed clustering run without refitting its clusters."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dinov3_cluster_projection import run_tsne_projection, save_umap_tsne_comparison


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def add_tsne(analysis: Path, perplexity: float = 30.0) -> dict:
    analysis = analysis.resolve()
    for name in ("tsne_coordinates.csv", "tsne_metadata.json", "kmeans_tsne.png", "ward_tsne.png",
                 "kmeans_umap_tsne_comparison.png"):
        if (analysis / name).exists():
            raise FileExistsError(f"t-SNE output already exists: {analysis / name}")
    metadata_path = analysis / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    projection_path = analysis / "projection_metadata.json"
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    selected = pd.read_csv(analysis / "clustered_valid_images.csv", encoding="utf-8-sig", float_precision="round_trip")
    standardized = pd.read_csv(analysis / "standardized_features.csv", encoding="utf-8-sig", float_precision="round_trip")
    existing = pd.read_csv(analysis / "projection_coordinates.csv", encoding="utf-8-sig", float_precision="round_trip")
    np.testing.assert_array_equal(selected["Filename"], standardized["Filename"])
    np.testing.assert_array_equal(selected["Filename"], existing["Filename"])
    features = metadata["features"]
    if list(standardized.columns) != ["Filename", *features]:
        raise ValueError("Standardized feature column order differs from the completed run")
    x = standardized[features].to_numpy(dtype=float)
    matrix_hash = sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest()
    if matrix_hash != projection["input_standardized_matrix_sha256"]:
        raise ValueError("Saved standardized matrix is not identical to the original projection input")
    labels = {method: selected[f"{method}_cluster"].to_numpy(dtype=int) for method in ("kmeans", "ward")}
    for method, values in labels.items():
        np.testing.assert_array_equal(values, existing[f"{method}_cluster"])
    protected = {str(p.relative_to(analysis)): digest(p) for p in analysis.rglob("*")
                 if p.is_file() and p.suffix in {".csv", ".npz", ".joblib", ".png", ".pdf", ".tiff"}}
    print(f"Computing t-SNE: {len(x)} rows, {x.shape[1]} original features, perplexity={perplexity:g}...", flush=True)
    info = run_tsne_projection(selected, x, features, labels, analysis,
                               seed=int(metadata["seed"]), perplexity=perplexity)
    table = pd.read_csv(analysis / "tsne_coordinates.csv", encoding="utf-8-sig", float_precision="round_trip")
    np.testing.assert_array_equal(table["Filename"], selected["Filename"])
    for method, values in labels.items():
        np.testing.assert_array_equal(table[f"{method}_cluster"], values)
    tsne_xy = table[["TSNE1", "TSNE2"]].to_numpy()
    if not np.isfinite(tsne_xy).all():
        raise RuntimeError("Invalid saved t-SNE coordinates")
    save_umap_tsne_comparison(existing[["UMAP1", "UMAP2"]].to_numpy(), tsne_xy,
                              labels["kmeans"], analysis / "kmeans_umap_tsne_comparison")
    for name, before in protected.items():
        if digest(analysis / name) != before:
            raise RuntimeError(f"Existing analysis artifact changed: {name}")
    info["added_at_utc"] = datetime.now(timezone.utc).isoformat()
    info["existing_analysis_artifacts_unchanged"] = True
    info["source_standardized_features_sha256"] = digest(analysis / "standardized_features.csv")
    info["code_sha256"] = {name: digest(Path(__file__).with_name(name)) for name in (
        "add_tsne_dinov3.py", "dinov3_cluster_projection.py", "dinov3_cluster_plot_style.py")}
    info["files"]["comparison"] = [f"kmeans_umap_tsne_comparison.{ext}" for ext in ("png", "pdf", "tiff")]
    (analysis / "tsne_metadata.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    projection["tsne"] = info
    projection["projection_axis_units"]["t-SNE"] = "dimensionless"
    projection["files"]["tsne_coordinates"] = "tsne_coordinates.csv"
    projection["files"]["figures"] += info["files"]["figures"] + info["files"]["comparison"]
    projection_path.write_text(json.dumps(projection, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["projection"] = projection
    metadata["tsne_metadata"] = "tsne_metadata.json"
    metadata["dimensionless_outputs"].append("t-SNE coordinates")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = {
        "sample_count": len(x), "input_matrix_identical_to_existing_projections": True,
        "kmeans_and_ward_labels_unchanged": True, "tsne_coordinates_finite": True,
        "existing_artifacts_unchanged": len(protected), "protected_sha256": protected,
        "parameters": info["parameters"], "kl_divergence": info["kl_divergence"],
    }
    (analysis / "tsne_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(x), "kl_divergence": info["kl_divergence"],
                      "existing_artifacts_unchanged": len(protected), "output": str(analysis)}, indent=2), flush=True)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()
    add_tsne(args.analysis_dir, args.perplexity)
