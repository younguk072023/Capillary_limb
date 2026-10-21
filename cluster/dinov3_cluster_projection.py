"""Common, unsupervised PCA/UMAP/t-SNE views of standardized diameter features.

Cluster labels are used only to color the figures after each projection is fit.
This module never fits a clustering algorithm or changes its input features.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from dinov3_cluster_plot_style import (
    CLUSTER_COLORS, PAPER_STYLE, SCATTER_ALPHA, SCATTER_EDGE_WIDTH, SCATTER_SIZE,
    plot_style_metadata, save_paper_figure, style_axes,
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _save_projection_plot(
    coordinates: np.ndarray,
    labels: np.ndarray,
    method: str,
    projection: str,
    output_dir: Path,
    axis_labels: tuple[str, str],
) -> None:
    unique_labels = np.unique(labels)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 5.1), constrained_layout=True)
        for cluster in unique_labels:
            mask = labels == cluster
            ax.scatter(
                coordinates[mask, 0], coordinates[mask, 1],
                color=CLUSTER_COLORS[int(cluster)],
                s=SCATTER_SIZE, alpha=SCATTER_ALPHA, linewidths=SCATTER_EDGE_WIDTH,
                label=f"Cluster {int(cluster)} (n = {int(mask.sum()):,})",
            )
        ax.set(
            xlabel=axis_labels[0], ylabel=axis_labels[1],
        )
        style_axes(ax)
        ax.legend(loc="best", fontsize=9)
        prefix = output_dir / f"{method}_{projection}"
        save_paper_figure(fig, prefix)


def save_umap_tsne_comparison(umap_xy: np.ndarray, tsne_xy: np.ndarray,
                              labels: np.ndarray, output: Path) -> None:
    """Title-free comparison using the same observations and saved labels."""
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
        for ax, coordinates, name in zip(axes, (umap_xy, tsne_xy), ("UMAP", "t-SNE")):
            for cluster in np.unique(labels):
                keep = labels == cluster
                ax.scatter(coordinates[keep, 0], coordinates[keep, 1],
                           s=SCATTER_SIZE, alpha=SCATTER_ALPHA, linewidths=SCATTER_EDGE_WIDTH,
                           color=CLUSTER_COLORS[int(cluster)],
                           label=f"Cluster {int(cluster)} (n = {int(keep.sum()):,})")
            ax.set(xlabel=f"{name} 1", ylabel=f"{name} 2")
            style_axes(ax)
            ax.legend(loc="best", fontsize=9)
        save_paper_figure(fig, output)


def run_tsne_projection(selected: pd.DataFrame, x_scaled: np.ndarray,
                        feature_cols: list[str] | tuple[str, ...],
                        labels_by_method: dict[str, np.ndarray], output_dir: Path,
                        seed: int = 42, perplexity: float = 30.0) -> dict:
    """Fit t-SNE on the original standardized features; labels only color plots."""
    x = np.asarray(x_scaled, dtype=np.float64)
    features = list(feature_cols)
    if x.shape != (len(selected), len(features)) or not np.isfinite(x).all():
        raise ValueError("t-SNE matrix must match the finite selected feature rows")
    if len(features) < 2 or not np.isfinite(perplexity) or not 0 < perplexity < len(x):
        raise ValueError("t-SNE needs at least two features and 0 < perplexity < sample count")
    if "Filename" not in selected or selected["Filename"].isna().any() or not selected["Filename"].is_unique:
        raise ValueError("t-SNE rows need unique nonmissing Filenames")
    if not labels_by_method:
        raise ValueError("No labels supplied to color t-SNE figures")
    for method, labels in labels_by_method.items():
        values = np.asarray(labels)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", method):
            raise ValueError("Invalid method name")
        if (values.shape != (len(x),) or not np.issubdtype(values.dtype, np.integer) or
                np.any(values < 0) or np.any(values >= len(CLUSTER_COLORS))):
            raise ValueError(f"Invalid {method} t-SNE coloring labels")
    config = {
        "n_components": 2, "perplexity": float(perplexity), "init": "pca",
        "learning_rate": "auto", "early_exaggeration": 12.0,
        "max_iter": 1000, "n_iter_without_progress": 300, "min_grad_norm": 1e-7,
        "metric": "euclidean", "random_state": int(seed),
        "method": "barnes_hut", "angle": 0.5, "n_jobs": 1,
    }
    estimator = TSNE(**config)
    with threadpool_limits(limits=1):
        coordinates = estimator.fit_transform(x)
    if coordinates.shape != (len(x), 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("t-SNE did not return finite 2D coordinates")
    if not np.isfinite(estimator.kl_divergence_):
        raise RuntimeError("Nonfinite t-SNE optimization objective")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = [name for name in ("Filename", "patient_id", "fold", "Status") if name in selected]
    table = selected[identity].reset_index(drop=True).copy()
    for method, labels in labels_by_method.items():
        table[f"{method}_cluster"] = np.asarray(labels)
        _save_projection_plot(coordinates, np.asarray(labels), method, "tsne", output_dir,
                              ("t-SNE 1", "t-SNE 2"))
    table["TSNE1"], table["TSNE2"] = coordinates[:, 0], coordinates[:, 1]
    table.to_csv(output_dir / "tsne_coordinates.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "purpose": "Visual exploration of saved clusters; not used for clustering or selecting K",
        "parameters": config, "fit_uses_cluster_labels": False,
        "sample_count": len(x), "input_dimension": len(features), "features_in_order": features,
        "input_units": "Dimensionless z-scores of the original physical features",
        "axis_units": "dimensionless", "shared_coordinates_across_methods": True,
        "input_standardized_matrix_sha256": sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest(),
        "threadpool_limit": 1, "iterations_completed": int(estimator.n_iter_) + 1,
        "kl_divergence": float(estimator.kl_divergence_),
        "effective_learning_rate": float(estimator.learning_rate_),
        "versions": {name: _package_version(name) for name in ("scikit-learn", "numpy", "scipy", "threadpoolctl")},
        "plot_format": plot_style_metadata(),
        "parameter_selection": "Fixed perplexity 30 by default; no selection based on visual separation",
        "interpretation_note": "2D overlap or separated islands do not establish cluster validity; t-SNE axes and inter-island distances have no physical unit",
        "files": {"coordinates": "tsne_coordinates.csv", "figures": [
            f"{method}_tsne.{extension}" for method in labels_by_method for extension in ("png", "pdf", "tiff")]},
    }
    (output_dir / "tsne_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def run_projections(
    selected: pd.DataFrame,
    x_scaled: np.ndarray,
    feature_cols: list[str] | tuple[str, ...],
    labels_by_method: dict[str, np.ndarray],
    output_dir: Path | str,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    tsne_perplexity: float = 30.0,
) -> dict:
    """Fit PCA, UMAP and t-SNE, sharing each set of coordinates across methods.

    ``x_scaled`` must contain the same standardized rows and feature columns used
    to fit the cluster models, in the positional order of ``selected``. The
    default five-feature analysis and the optional three-feature sensitivity
    analysis are supported. Output labels retain their zero-based CSV values.
    """
    # Import lazily so the main script can provide a useful dependency check.
    from umap import UMAP

    x = np.asarray(x_scaled, dtype=np.float64)
    features = list(feature_cols)
    if x.ndim != 2 or x.shape != (len(selected), len(features)):
        raise ValueError("Projection matrix shape must match selected rows and feature columns")
    if len(selected) < 4 or len(features) < 2 or not np.isfinite(x).all():
        raise ValueError("Projections need at least four rows, two features, and finite values")
    if len(set(features)) != len(features):
        raise ValueError("Projection feature names must be unique")
    if not np.any(np.std(x, axis=0) > 0):
        raise ValueError("Projection features have no variance")
    if "Filename" not in selected or selected["Filename"].isna().any():
        raise ValueError("Each projection row must have a Filename")
    if not selected["Filename"].is_unique:
        raise ValueError("Projection Filenames must be unique")
    if (not isinstance(n_neighbors, (int, np.integer)) or
            not 2 <= n_neighbors < len(selected)):
        raise ValueError("UMAP n_neighbors must be an integer from 2 to sample count minus 1")
    if not np.isfinite(min_dist) or not 0 <= min_dist <= 1:
        raise ValueError("UMAP min_dist must be between 0 and the fixed spread of 1")
    if not labels_by_method:
        raise ValueError("At least one set of cluster labels is required")
    if not np.isfinite(tsne_perplexity) or not 0 < tsne_perplexity < len(selected):
        raise ValueError("Need 0 < t-SNE perplexity < sample count")
    validated_labels = {}
    for method, labels in labels_by_method.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", method):
            raise ValueError(f"Invalid projection method name: {method!r}")
        values = np.asarray(labels)
        if values.shape != (len(selected),):
            raise ValueError(f"{method} labels must match the number and order of selected rows")
        if (not np.issubdtype(values.dtype, np.integer) or
                np.any(values < 0) or np.any(values >= len(CLUSTER_COLORS))):
            raise ValueError(f"{method} labels must be integers in [0, {len(CLUSTER_COLORS) - 1}]")
        validated_labels[method] = values.astype(int, copy=False)

    # Full PCA records the variance of every original dimension. Only PC1/PC2
    # are plotted; no cluster labels enter either projection fit.
    pca = PCA(n_components=min(x.shape), svd_solver="full")
    pca_full = pca.fit_transform(x)
    pca_xy = pca_full[:, :2]
    umap_config = {
        "n_components": 2,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "spread": 1.0,
        "metric": metric,
        "init": "spectral",
        "random_state": int(seed),
        "transform_seed": int(seed),
        "n_jobs": 1,
    }
    reducer = UMAP(**umap_config)
    umap_xy = reducer.fit_transform(x)
    for name, coordinates in (("PCA", pca_xy), ("UMAP", umap_xy)):
        if coordinates.shape != (len(selected), 2) or not np.isfinite(coordinates).all():
            raise RuntimeError(f"{name} did not produce finite two-dimensional coordinates")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identifier_cols = [c for c in ("Filename", "patient_id", "fold", "Status") if c in selected]
    coordinates = selected[identifier_cols].reset_index(drop=True).copy()
    for method, labels in validated_labels.items():
        coordinates[f"{method}_cluster"] = labels
    coordinates["PC1"] = pca_xy[:, 0]
    coordinates["PC2"] = pca_xy[:, 1]
    coordinates["UMAP1"] = umap_xy[:, 0]
    coordinates["UMAP2"] = umap_xy[:, 1]
    coordinates.to_csv(output_dir / "projection_coordinates.csv", index=False, encoding="utf-8-sig")
    components = pd.DataFrame(pca.components_, columns=features)
    components.insert(0, "component", [f"PC{i + 1}" for i in range(len(components))])
    components.to_csv(output_dir / "pca_components.csv", index=False, encoding="utf-8-sig")
    variance = pd.DataFrame({
        "component": components["component"],
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        "singular_value": pca.singular_values_,
    })
    variance.to_csv(output_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    axis_labels = tuple(
        f"PC{i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}%)" for i in range(2)
    )
    for method, labels in validated_labels.items():
        _save_projection_plot(pca_xy, labels, method, "pca", output_dir, axis_labels)
        _save_projection_plot(umap_xy, labels, method, "umap", output_dir, ("UMAP 1", "UMAP 2"))

    tsne_info = run_tsne_projection(selected, x, features, validated_labels, output_dir,
                                    seed=seed, perplexity=tsne_perplexity)
    if "kmeans" in validated_labels:
        tsne_xy = pd.read_csv(output_dir / "tsne_coordinates.csv", float_precision="round_trip")[["TSNE1", "TSNE2"]].to_numpy()
        save_umap_tsne_comparison(umap_xy, tsne_xy, validated_labels["kmeans"],
                                  output_dir / "kmeans_umap_tsne_comparison")

    metadata = {
        "purpose": "Visual exploration only; cluster models and K scores use the original standardized features",
        "fit_uses_cluster_labels": False,
        "shared_coordinates_across_methods": True,
        "sample_count": int(x.shape[0]),
        "input_dimension": int(x.shape[1]),
        "output_dimension": 2,
        "features_in_order": features,
        "feature_units_before_standardization": {
            feature: "\u00b5m" if feature.endswith("_um") else
            "px" if feature.endswith("_px") else "dimensionless"
            for feature in features
        },
        "input_matrix_units": "Dimensionless z-scores after feature-wise standardization",
        "projection_axis_units": {"PCA": "dimensionless", "UMAP": "dimensionless", "t-SNE": "dimensionless"},
        "input_standardized_matrix_sha256": sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest(),
        "row_order": "Same positional row order as selected measurements and projection_coordinates.csv",
        "pca": {
            "svd_solver": "full",
            "fitted_components": int(pca.n_components_),
            "plotted_components": [1, 2],
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]),
            "pc1_pc2_cumulative_explained_variance_ratio": float(pca.explained_variance_ratio_[:2].sum()),
        },
        "umap": {"parameters": umap_config, "effective_n_neighbors": int(reducer._n_neighbors)},
        "tsne": tsne_info,
        "versions": {name: _package_version(name) for name in (
            "umap-learn", "scikit-learn", "numpy", "scipy", "numba", "pynndescent", "matplotlib",
        )},
        "cluster_colors": {
            str(int(label)): CLUSTER_COLORS[int(label)]
            for label in np.unique(np.concatenate(list(validated_labels.values())))
        },
        "cluster_label_note": "Colors follow the supplied zero-based cluster IDs, sorted by mean diameter within each method",
        "plot_format": plot_style_metadata(),
        "files": {
            "coordinates": "projection_coordinates.csv",
            "tsne_coordinates": "tsne_coordinates.csv",
            "pca_components": "pca_components.csv",
            "pca_variance": "pca_explained_variance.csv",
            "figures": [f"{method}_{projection}.{suffix}"
                        for method in validated_labels
                        for projection in ("pca", "umap", "tsne") for suffix in ("png", "pdf", "tiff")]
                       + ([f"kmeans_umap_tsne_comparison.{suffix}" for suffix in ("png", "pdf", "tiff")]
                          if "kmeans" in validated_labels else []),
        },
    }
    (output_dir / "projection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return metadata
