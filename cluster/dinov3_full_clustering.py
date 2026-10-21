"""Cluster full-cohort DINOv3 OOF limb measurements.

The input is the canonical CSV from export_dinov3_full_measurements.py.  Every
image remains in the assignment table; rows without finite positive measurements
receive no cluster. Measured rows with Uncertain QC are included by default.  The width-ranked thin/wide branches are *proxies*, not
anatomically identified artery/vein labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from dinov3_cluster_projection import run_projections
from dinov3_cluster_interpretation import run_interpretation
from dinov3_cluster_plot_style import PAPER_STYLE, CLUSTER_COLORS, plot_style_metadata, save_paper_figure, style_axes


REQUIRED = (
    "Filename", "patient_id", "fold", "Status", "Reason", "Run_ID",
    "Algorithm_SHA256", "Predicted_Mask", "Mask_SHA256",
    "Loop_Diameter_px", "Estimated_Arterial_Diameter_px",
    "Estimated_Venous_Diameter_px",
)
PHYSICAL_PX = (
    "thin_branch_diameter_px", "wide_branch_diameter_px", "loop_diameter_px",
)
PHYSICAL = tuple(name.removesuffix("_px") + "_um" for name in PHYSICAL_PX)
FEATURES = (*PHYSICAL, "wide_to_thin_ratio", "mean_diameter_um")
FEATURE_SETS = {"legacy5": FEATURES, "core3": PHYSICAL}
SOURCE_COLUMNS = dict(zip(PHYSICAL_PX, (
    "Estimated_Arterial_Diameter_px", "Estimated_Venous_Diameter_px",
    "Loop_Diameter_px",
)))
SEED = 42
DEFAULT_UM_PER_PIXEL = 0.55
CALIBRATION_REFERENCE = (
    "C:/Users/USER/Desktop/Multi-Task-Learning/outputs/"
    "predicted_mask_measurements_dinov3_141_20261008_run3/provenance.json"
)


def check_analysis_dependencies() -> dict:
    """Fail before creating output if either requested analysis is unavailable."""
    for module, package in (("umap", "umap-learn"), ("shap", "shap")):
        try:
            import_module(module)
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import {package} in {sys.executable}. "
                "Install requirements_dinov3_clustering.txt in this environment."
            ) from exc
    return {name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "umap-learn", "shap", "numba", "llvmlite", "pynndescent",
    )}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input(path: Path, policy: str, expected_count: int,
               expected_run_id: str, um_per_pixel: float = DEFAULT_UM_PER_PIXEL) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError("um_per_pixel must be finite and positive")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"Filename": str})
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")
    if frame.empty or frame["Filename"].isna().any() or not frame["Filename"].is_unique:
        raise ValueError("Filename must be present and unique for every image")
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} images, found {len(frame)}")
    if frame["Run_ID"].nunique(dropna=False) != 1:
        raise ValueError("Input mixes DINOv3 runs")
    if str(frame["Run_ID"].iloc[0]) != expected_run_id:
        raise ValueError(f"Unexpected DINOv3 run ID: {frame['Run_ID'].iloc[0]}")
    if frame["Algorithm_SHA256"].nunique(dropna=False) != 1:
        raise ValueError("Input mixes measurement algorithm versions")
    if not set(frame["Status"].dropna().unique()).issubset({"Success", "Uncertain", "Failed"}):
        raise ValueError("Unexpected analyzer Status")
    frame = frame.sort_values("Filename", kind="stable").reset_index(drop=True)
    for feature, source in SOURCE_COLUMNS.items():
        frame[feature] = pd.to_numeric(frame[source], errors="coerce")
    # Mean of three sites requires all three measurements, including Failed rows.
    frame["mean_diameter_px"] = frame[list(PHYSICAL_PX)].mean(axis=1, skipna=False)
    # Convert every measured length; leave image coordinates and point counts in
    # their original pixel/index frame so saved overlays remain valid.
    for column in list(frame.columns):
        if column.endswith("_px"):
            frame[column.removesuffix("_px") + "_um"] = (
                pd.to_numeric(frame[column], errors="coerce") * um_per_pixel
            )
    valid_physical = np.isfinite(frame[list(PHYSICAL)].to_numpy()).all(axis=1)
    valid_physical &= (frame[list(PHYSICAL)].to_numpy() > 0).all(axis=1)
    frame["wide_to_thin_ratio"] = (
        frame["wide_branch_diameter_px"] / frame["thin_branch_diameter_px"]
    )
    valid_features = valid_physical & np.isfinite(frame[list(FEATURES)].to_numpy()).all(axis=1)
    accepted = {"Success"} if policy == "success" else {"Success", "Uncertain"}
    frame["inclusion_reason"] = np.select(
        [~frame["Status"].isin(accepted), ~valid_features],
        ["qc_" + frame["Status"].fillna("missing"), "nonpositive_or_nonfinite_measurement"],
        default="included",
    )
    selected = frame.loc[frame["inclusion_reason"].eq("included")].copy()
    if len(selected) < 9:
        raise ValueError(f"Only {len(selected)} usable measurements")
    if (selected["wide_branch_diameter_px"] < selected["thin_branch_diameter_px"]).any():
        raise ValueError("Width-ranked branch order is inconsistent")
    return frame, selected


def check_provenance(path: Path, expected_count: int, expected_run_id: str,
                     allow_unverified: bool) -> dict | None:
    source = path.parent / "provenance.json"
    if not source.is_file():
        if allow_unverified:
            return None
        raise ValueError(f"Measurement provenance is missing: {source}")
    provenance = json.loads(source.read_text(encoding="utf-8"))
    if (provenance.get("csv_sha256") != file_hash(path) or
            provenance.get("rows") != expected_count or
            provenance.get("run_id") != expected_run_id or
            provenance.get("mask_source") != "Each image's saved outer-test predicted mask"):
        raise ValueError("Measurement provenance does not match this full OOF CSV")
    return provenance


def relabel_by_mean_size(labels: np.ndarray, mean_size: np.ndarray) -> np.ndarray:
    order = sorted(np.unique(labels), key=lambda label: (mean_size[labels == label].mean(), int(label)))
    mapping = {raw: stable for stable, raw in enumerate(order)}
    return np.array([mapping[value] for value in labels], dtype=int)


def fit_model(method: str, k: int, x: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, object]:
    if method == "kmeans":
        model = KMeans(n_clusters=k, init="k-means++", random_state=SEED, n_init=20)
    elif method == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        raise ValueError(method)
    raw = model.fit_predict(x)
    return relabel_by_mean_size(raw, sizes), model


def score_row(method: str, k: int, x: np.ndarray, labels: np.ndarray, model: object) -> dict:
    counts = np.bincount(labels, minlength=k)
    if method == "kmeans":
        # With tolerance-based convergence, recomputed label means can differ
        # slightly from the fitted centers. Verify inertia against fitted centers.
        measured_sse = float(np.square(x - model.cluster_centers_[model.labels_]).sum())
        sse = float(model.inertia_)
    else:
        measured_sse = sum(float(np.square(x[labels == cluster] - x[labels == cluster].mean(axis=0)).sum())
                           for cluster in range(k))
        sse = measured_sse
    if not np.isclose(sse, measured_sse, rtol=1e-9, atol=1e-8):
        raise RuntimeError("Model SSE disagrees with the assigned clusters")
    return {
        "method": method, "k": k,
        "sse": sse,
        "silhouette": float(silhouette_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "smallest_cluster_n": int(counts.min()),
        "largest_cluster_n": int(counts.max()),
    }


def plot_scores(scores: pd.DataFrame, output: Path, final_k: int) -> None:
    with plt.rc_context(PAPER_STYLE):
        for metric, filename, ylabel in (
            ("silhouette", "silhouette_by_k", "Silhouette score"),
            ("sse", "elbow_by_k", "SSE in standardized feature space"),
        ):
            fig, ax = plt.subplots(figsize=(5.8, 5.1), constrained_layout=True)
            for method, group in scores.groupby("method"):
                if method != "kmeans":
                    continue
                ax.plot(group["k"], group[metric], marker="o", markersize=4.5,
                        color=CLUSTER_COLORS[0 if method == "kmeans" else 1], linewidth=1.2,
                        label={"kmeans": "K-means", "ward": "Ward"}[method])
            ax.set(xlabel="Number of clusters (K)", ylabel=ylabel)
            ax.set_xticks(sorted(scores["k"].unique()))
            style_axes(ax)
            save_paper_figure(fig, output / filename)


def summarize(selected: pd.DataFrame, method: str, column: str) -> pd.DataFrame:
    rows = []
    for cluster, group in selected.groupby(column, sort=True):
        row = {"method": method, "cluster": int(cluster), "n_images": len(group),
               "n_patients": group["patient_id"].nunique()}
        for feature in FEATURES:
            row[f"{feature}_mean"] = group[feature].mean()
            row[f"{feature}_median"] = group[feature].median()
            row[f"{feature}_sd"] = group[feature].std()
        rows.append(row)
    return pd.DataFrame(rows)


def patient_composition(selected: pd.DataFrame, method: str, column: str, k: int) -> pd.DataFrame:
    counts = pd.crosstab(selected["patient_id"], selected[column]).reindex(columns=range(k), fill_value=0)
    total = counts.sum(axis=1)
    rows = []
    for patient in counts.index:
        for cluster in range(k):
            value = int(counts.loc[patient, cluster])
            rows.append({"method": method, "patient_id": patient, "cluster": cluster,
                         "image_count": value, "patient_total_images": int(total.loc[patient]),
                         "within_patient_fraction": value / total.loc[patient]})
    return pd.DataFrame(rows)


def representatives(selected: pd.DataFrame, x: np.ndarray, column: str, method: str, k: int) -> pd.DataFrame:
    rows = []
    for cluster in range(k):
        indices = np.flatnonzero(selected[column].to_numpy() == cluster)
        center = x[indices].mean(axis=0)
        distances = np.linalg.norm(x[indices] - center, axis=1)
        for rank, local in enumerate(np.argsort(distances)[:5], start=1):
            item = selected.iloc[int(indices[local])]
            rows.append({"method": method, "cluster": cluster, "rank": rank,
                         "Filename": item["Filename"], "patient_id": item["patient_id"],
                         "Predicted_Mask": item["Predicted_Mask"],
                         "scaled_distance_to_cluster_mean": float(distances[local]),
                         **{feature: item[feature] for feature in FEATURES}})
    return pd.DataFrame(rows)


def run(input_csv: Path, output_dir: Path, k: int, max_k: int, policy: str,
        feature_set: str, expected_count: int, expected_run_id: str,
        allow_unverified: bool, umap_neighbors: int = 15, umap_min_dist: float = 0.1,
        um_per_pixel: float = DEFAULT_UM_PER_PIXEL, tsne_perplexity: float = 30.0) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    provenance = check_provenance(input_csv, expected_count, expected_run_id,
                                  allow_unverified)
    all_rows, selected = load_input(input_csv, policy, expected_count, expected_run_id, um_per_pixel)
    n = len(selected)
    if not 2 <= k <= max_k < n:
        raise ValueError(f"Need 2 <= k <= max_k < usable sample count ({n})")
    if max_k > 12:
        raise ValueError("max_k > 12 would produce an overly broad search")
    if not 2 <= umap_neighbors < n or not 0 <= umap_min_dist <= 1:
        raise ValueError("Need 2 <= umap-neighbors < usable count and 0 <= umap-min-dist <= 1")
    if not np.isfinite(tsne_perplexity) or not 0 < tsne_perplexity < n:
        raise ValueError("Need 0 < tsne-perplexity < usable sample count")
    print("Checking PCA/UMAP and RF/SHAP dependencies...", flush=True)
    package_versions = check_analysis_dependencies()
    feature_cols = FEATURE_SETS[feature_set]
    x = selected[list(feature_cols)].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    sizes = selected["mean_diameter_um"].to_numpy()
    scores, final_models = [], {}
    for method in ("kmeans", "ward"):
        for candidate in range(2, max_k + 1):
            labels, model = fit_model(method, candidate, x_scaled, sizes)
            scores.append(score_row(method, candidate, x_scaled, labels, model))
            print(f"{method} K={candidate}: silhouette={scores[-1]['silhouette']:.4f}, "
                  f"SSE={scores[-1]['sse']:.3f}", flush=True)
            if candidate == k:
                final_models[method] = (labels, model)
    scores_df = pd.DataFrame(scores)
    scores_df["sse_reduction_from_previous_k"] = -scores_df.groupby("method")["sse"].diff()
    scores_df["sse_reduction_fraction_from_previous_k"] = (
        scores_df["sse_reduction_from_previous_k"] / scores_df.groupby("method")["sse"].shift()
    )
    for method, (labels, _model) in final_models.items():
        selected[f"{method}_cluster"] = labels
    selected["cluster"] = selected["kmeans_cluster"]
    all_rows = all_rows.merge(
        selected[["Filename", "cluster", "kmeans_cluster", "ward_cluster"]],
        on="Filename", how="left", validate="one_to_one",
    )
    if len(all_rows) != len(pd.read_csv(input_csv, encoding="utf-8-sig")):
        raise RuntimeError("The assignment table lost images")
    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows.to_csv(output_dir / "cluster_assignments_all.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "clustered_valid_images.csv", index=False, encoding="utf-8-sig")
    scores_df.to_csv(output_dir / "cluster_k_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols, "mean": scaler.mean_, "scale": scaler.scale_,
                  "source_unit": ["µm" if name.endswith("_um") else "dimensionless" for name in feature_cols]}).to_csv(
        output_dir / "scaler_parameters.csv", index=False, encoding="utf-8-sig")
    scaled_frame = pd.DataFrame(x_scaled, columns=feature_cols)
    scaled_frame.insert(0, "Filename", selected["Filename"].to_numpy())
    scaled_frame.to_csv(output_dir / "standardized_features.csv", index=False, encoding="utf-8-sig")
    all_summaries, all_patient, all_representatives = [], [], []
    for method, (labels, model) in final_models.items():
        column = f"{method}_cluster"
        all_summaries.append(summarize(selected, method, column))
        all_patient.append(patient_composition(selected, method, column, k))
        all_representatives.append(representatives(selected, x_scaled, column, method, k))
        if method == "kmeans":
            centers = np.vstack([
                model.cluster_centers_[model.labels_[np.flatnonzero(labels == cluster)[0]]]
                for cluster in range(k)
            ])
            pd.DataFrame(centers, columns=feature_cols).assign(cluster=range(k)).to_csv(
                output_dir / "kmeans_centers_scaled.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_summaries, ignore_index=True).to_csv(
        output_dir / "cluster_feature_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_patient, ignore_index=True).to_csv(
        output_dir / "patient_cluster_composition.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_representatives, ignore_index=True).to_csv(
        output_dir / "representative_images.csv", index=False, encoding="utf-8-sig")
    plot_scores(scores_df, output_dir, k)
    print("Projecting the standardized features with PCA, UMAP and t-SNE...", flush=True)
    projection_info = run_projections(
        selected, x_scaled, feature_cols,
        {method: labels for method, (labels, _) in final_models.items()},
        output_dir, seed=SEED, n_neighbors=umap_neighbors, min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )
    print("Computing Kruskal-Wallis, effect sizes, RF surrogate and SHAP...", flush=True)
    interpretation_info = run_interpretation(
        selected, feature_cols, final_models["kmeans"][0], output_dir / "interpretation", seed=SEED,
    )
    kmeans_scores = scores_df.loc[scores_df["method"].eq("kmeans")]
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "python_executable": sys.executable,
        "package_versions": package_versions,
        "analysis_code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in (
            "dinov3_full_clustering.py", "dinov3_cluster_projection.py", "dinov3_cluster_interpretation.py",
            "dinov3_cluster_plot_style.py",
        )},
        "plot_format": plot_style_metadata(),
        "input_csv": str(input_csv), "input_sha256": file_hash(input_csv),
        "run_id": str(all_rows["Run_ID"].iloc[0]),
        "algorithm_sha256": str(all_rows["Algorithm_SHA256"].iloc[0]),
        "image_count": len(all_rows), "included_count": n,
        "qc_counts": dict(Counter(all_rows["Status"].fillna("missing"))),
        "inclusion_counts": dict(Counter(all_rows["inclusion_reason"])),
        "qc_policy": policy, "k": k, "max_k_evaluated": max_k,
        "seed": SEED, "kmeans_init": "k-means++", "kmeans_n_init": 20, "ward_linkage": "ward",
        "k_selection": {
            "selected_k": k, "selection": "User-configurable final K; not chosen by UMAP, PCA or t-SNE",
            "range_evaluated": [2, max_k],
            "highest_silhouette_k": int(kmeans_scores.loc[kmeans_scores["silhouette"].idxmax(), "k"]),
            "elbow": "SSE curve for visual review; no automatic elbow optimum is asserted",
        },
        "feature_set": feature_set, "features": list(feature_cols),
        "feature_units": {name: "µm" if name.endswith("_um") else "dimensionless" for name in feature_cols},
        "calibration": {
            "um_per_pixel": float(um_per_pixel), "raw_unit": "px", "diameter_unit": "µm",
            "source": "Reused prior user-specified calibration" if um_per_pixel == DEFAULT_UM_PER_PIXEL else "CLI --um-per-pixel override",
            "reference_provenance": CALIBRATION_REFERENCE if um_per_pixel == DEFAULT_UM_PER_PIXEL else None,
            "reference_provenance_sha256": file_hash(Path(CALIBRATION_REFERENCE))
                if um_per_pixel == DEFAULT_UM_PER_PIXEL and Path(CALIBRATION_REFERENCE).is_file() else None,
            "coordinate_frame": provenance.get("coordinate_frame") if provenance else None,
            "conversion": "Every measured *_px length multiplied by um_per_pixel into corresponding *_um; raw px retained",
            "mean_diameter": "All three site diameters required; incomplete rows retain missing mean",
        },
        "dimensionless_outputs": ["wide_to_thin_ratio", "standardized_features.csv", "kmeans_centers_scaled.csv",
                                  "PCA coordinates", "UMAP coordinates", "t-SNE coordinates", "silhouette", "standardized SSE",
                                  "rank epsilon-squared", "SHAP probability contributions"],
        "source_provenance": str(input_csv.parent / "provenance.json") if provenance else None,
        "feature_meaning": {
            "thin_branch_diameter_um": "Width-ranked thinner branch in micrometers; anatomical arterial identity unconfirmed",
            "wide_branch_diameter_um": "Width-ranked wider branch in micrometers; anatomical venous identity unconfirmed",
            "loop_diameter_um": "DINOv3 predicted U-D Euclidean distance in micrometers",
            "wide_to_thin_ratio": "wide / thin",
            "mean_diameter_um": "arithmetic mean of thin, wide, and loop in micrometers",
        },
        "cluster_label_rule": "Cluster numbers sorted by mean_diameter_um ascending within each method",
        "metric_note": "K scores are descriptive; post-cluster tests on the same features are not independent validation",
        "pca_explained_variance": {method: {key: projection_info["pca"][key] for key in ("pc1", "pc2")}
                                   for method in final_models},
        "projection": projection_info,
        "interpretation": interpretation_info,
        "kmeans_vs_ward_adjusted_rand_index": float(adjusted_rand_score(
            selected["kmeans_cluster"], selected["ward_cluster"])),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("image_count", "included_count", "qc_counts",
                  "inclusion_counts", "kmeans_vs_ward_adjusted_rand_index")}, indent=2), flush=True)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Canonical full OOF measurements CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for cluster results")
    parser.add_argument("--k", type=int, default=3, help="Prespecified final K (default: 3)")
    parser.add_argument("--max-k", type=int, default=6, help="Largest K to score (default: 6; score K=2..6)")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP neighborhood size (default: 15)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP minimum distance (default: 0.1)")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0,
                        help="t-SNE perplexity for visualization (default: 30)")
    parser.add_argument("--um-per-pixel", type=float, default=DEFAULT_UM_PER_PIXEL,
                        help="Micrometers per source-frame pixel (default: 0.55, existing calibration)")
    parser.add_argument("--qc-policy", choices=("success", "measured"), default="measured",
                        help="measured (default) includes finite positive Uncertain; success is a QC sensitivity analysis")
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="legacy5",
                        help="legacy5 reproduces the earlier five-variable design; core3 uses direct measurements")
    parser.add_argument("--expected-count", type=int, default=1397)
    parser.add_argument("--expected-run-id", default="dbc65ae36d7e")
    parser.add_argument("--allow-unverified-input", action="store_true",
                        help="Allow a CSV without the matching OOF export provenance")
    args = parser.parse_args(argv)
    run(args.input.resolve(), args.output_dir.resolve(), args.k, args.max_k,
        args.qc_policy, args.feature_set, args.expected_count,
        args.expected_run_id, args.allow_unverified_input, args.umap_neighbors, args.umap_min_dist,
        args.um_per_pixel, args.tsne_perplexity)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
