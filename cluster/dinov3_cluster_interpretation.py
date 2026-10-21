"""Descriptive statistics and RF/SHAP interpretation of existing K-means labels.

The clustering variables are reused for interpretation, so Kruskal-Wallis results
describe the derived groups rather than independently validate their existence.
RF performance is resubstitution fidelity to those labels, not generalization or
clinical classification performance. Width-ranked branches remain proxies.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import kruskal
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dinov3_cluster_plot_style import (
    CLUSTER_COLORS,
    PAPER_STYLE,
    SCATTER_ALPHA,
    SCATTER_EDGE_WIDTH,
    SCATTER_SIZE,
    plot_style_metadata,
    save_paper_figure,
    style_axes,
)


FEATURE_LABELS = {
    "thin_branch_diameter_um": "Arterial diameter (\u00b5m)",
    "wide_branch_diameter_um": "Venous diameter (\u00b5m)",
    "loop_diameter_um": "Loop diameter (\u00b5m)",
    "mean_diameter_um": "Mean diameter (\u00b5m)",
    "thin_branch_diameter_px": "Arterial diameter (px)",
    "wide_branch_diameter_px": "Venous diameter (px)",
    "loop_diameter_px": "Loop diameter (px)",
    "wide_to_thin_ratio": "Venous / arterial diameter ratio",
    "mean_diameter_px": "Mean diameter (px)",
}


def _feature_units(feature_cols: list | tuple) -> dict[str, str]:
    """Describe supplied raw features; projections and SHAP retain their own units."""
    return {
        feature: "\u00b5m" if feature.endswith("_um") else
        "px" if feature.endswith("_px") else "dimensionless"
        for feature in feature_cols
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _normalize_shap_values(raw: object, n: int, p: int, k: int) -> np.ndarray:
    """Normalize documented SHAP formats to [sample, feature, model class].

    SHAP <0.45 returns one [sample, feature] array per class; newer releases
    return [sample, feature, class]. Never guess/reorder the class axis.
    """
    if isinstance(raw, (list, tuple)):
        if len(raw) != k:
            raise ValueError(f"SHAP returned {len(raw)} class arrays, expected {k}")
        arrays = [np.asarray(value, dtype=float) for value in raw]
        if any(value.shape != (n, p) for value in arrays):
            raise ValueError(f"Each SHAP class array must have shape {(n, p)}")
        values = np.stack(arrays, axis=2)
    else:
        values = np.asarray(raw, dtype=float)
    if values.shape != (n, p, k):
        raise ValueError(f"SHAP shape {values.shape}; expected sample/feature/class {(n, p, k)}")
    if not np.isfinite(values).all():
        raise ValueError("SHAP returned nonfinite attributions")
    return values


def _statistics(x: pd.DataFrame, labels: np.ndarray, classes: np.ndarray,
                output: Path) -> dict:
    n, p = x.shape
    k = len(classes)
    units = _feature_units(list(x.columns))
    stats_rows = []
    descriptives = []
    for feature in x.columns:
        groups = [x.loc[labels == label, feature].to_numpy() for label in classes]
        if np.unique(x[feature].to_numpy()).size == 1:
            h_stat, p_value, status = 0.0, 1.0, "constant_feature_no_difference"
        else:
            result = kruskal(*groups, nan_policy="raise")
            h_stat, p_value, status = float(result.statistic), float(result.pvalue), "computed"
        adjusted = float((h_stat - k + 1) / (n - k)) if n > k else None
        stats_rows.append({
            "feature": feature,
            "feature_unit": units[feature],
            "n": n,
            "k": k,
            "H_statistic": h_stat,
            "p_value": p_value,
            "p_bonferroni_across_features": min(1.0, p_value * p),
            "epsilon_squared_rank_H_over_N_minus_1": h_stat / (n - 1),
            "legacy_adjusted_H_minus_k_plus_1_over_N_minus_k": adjusted,
            "legacy_adjusted_clipped_0_1": float(np.clip(adjusted, 0, 1)) if adjusted is not None else None,
            "test_status": status,
            **{f"cluster_{label}_n": len(group) for label, group in zip(classes, groups)},
        })
        for label, group in zip(classes, groups):
            descriptives.append({
                "feature": feature, "feature_unit": units[feature],
                "cluster": int(label), "n": len(group),
                "mean": float(np.mean(group)),
                "std": float(np.std(group, ddof=1)) if len(group) > 1 else None,
                "median": float(np.median(group)),
                "q1": float(np.quantile(group, 0.25)),
                "q3": float(np.quantile(group, 0.75)),
                "min": float(np.min(group)), "max": float(np.max(group)),
            })

    pd.DataFrame(stats_rows).to_csv(output / "kruskal_wallis_effect_sizes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(descriptives).to_csv(output / "feature_descriptives_by_cluster.csv", index=False, encoding="utf-8-sig")

    return {
        "test": "scipy.stats.kruskal; tie-corrected H; chi-square p-value",
        "primary_effect_size_column": "epsilon_squared_rank_H_over_N_minus_1",
        "primary_effect_size_formula": "H / (N - 1)",
        "primary_effect_size_legacy_source": "stat_statistics.py",
        "alternative_legacy_formula": "(H - k + 1) / (N - k)",
        "alternative_effect_size_name": "Rank eta-squared; called epsilon-squared in legacy K-means.py",
        "alternative_legacy_source": "K-means.py; both raw and clipped [0,1] values retained",
        "p_value_adjustment": f"Bonferroni across {p} feature tests; raw p-values also retained",
        "smallest_cluster_n": int(min(np.sum(labels == label) for label in classes)),
        "purpose": "Descriptive differences in variables used to construct the clusters; not independent validation",
        "rows": stats_rows,
    }


def _feature_boxplots(x: pd.DataFrame, labels: np.ndarray,
                      classes: np.ndarray, output: Path) -> None:
    ncols = 3
    nrows = int(np.ceil(x.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.2 * nrows), squeeze=False)
    for ax, feature in zip(axes.flat, x.columns):
        groups = [x.loc[labels == label, feature].to_numpy() for label in classes]
        boxes = ax.boxplot(
            groups, patch_artist=True, widths=0.6,
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            medianprops={"linewidth": 1.0, "color": "black"},
            flierprops={"markersize": 1.5, "alpha": 0.4},
        )
        for patch, label in zip(boxes["boxes"], classes):
            patch.set_facecolor(CLUSTER_COLORS[int(label)])
            patch.set_alpha(0.65)
        ax.set_xticks(np.arange(1, len(classes) + 1))
        ax.set_xticklabels([f"Cluster {label}\n(n={np.sum(labels == label)})" for label in classes])
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
        style_axes(ax)
    for ax in list(axes.flat)[x.shape[1]:]:
        ax.set_visible(False)
    fig.tight_layout()
    save_paper_figure(fig, output / "feature_boxplots")


def _confusion_plot(matrix: np.ndarray, classes: np.ndarray, accuracy: float,
                    output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    heatmap = ax.imshow(matrix, cmap="Blues")
    colorbar = fig.colorbar(heatmap, ax=ax, label="Number of capillaries")
    colorbar.ax.tick_params(labelsize=9)
    colorbar.ax.yaxis.label.set_size(11)
    positions = np.arange(len(classes))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels([f"Cluster {label}" for label in classes])
    ax.set_yticklabels([f"Cluster {label}" for label in classes])
    for i in positions:
        for j in positions:
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
    ax.set(xlabel="RF surrogate prediction (training data)", ylabel="K-means pseudo-target")
    style_axes(ax)
    ax.grid(False)
    fig.tight_layout()
    save_paper_figure(fig, output / "rf_training_confusion_matrix")


@plt.rc_context(PAPER_STYLE)
def render_interpretation_figures(
    x: pd.DataFrame, labels: np.ndarray, classes: np.ndarray,
    values: np.ndarray, matrix: np.ndarray, accuracy: float,
    output: Path, seed: int = 42,
) -> None:
    """Render saved features, SHAP values and RF counts without fitting models.

    Array rows and class order must match the saved interpretation outputs.
    This function only writes figures; it does not change analysis tables.
    """
    import shap

    output = Path(output)
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    values = np.asarray(values)
    if values.shape != (len(x), x.shape[1], len(classes)):
        raise ValueError("Plot SHAP axes must be [sample, feature, class]")
    if labels.shape != (len(x),) or np.asarray(matrix).shape != (len(classes), len(classes)):
        raise ValueError("Plot labels and confusion matrix must match feature rows and classes")
    output.mkdir(parents=True, exist_ok=True)
    _feature_boxplots(x, labels, classes, output)
    _confusion_plot(matrix, classes, accuracy, output)

    mean_abs = np.abs(values).mean(axis=0)
    feature_cols = list(x.columns)
    order = np.argsort(mean_abs.mean(axis=1))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.85 * len(feature_cols))))
    positions = np.arange(len(feature_cols), dtype=float)
    height = 0.8 / len(classes)
    for class_idx, label in enumerate(classes):
        offset = (class_idx - (len(classes) - 1) / 2) * height
        ax.barh(positions + offset, mean_abs[order, class_idx], height=height,
                label=f"Cluster {label}", color=CLUSTER_COLORS[int(label)])
    ax.set_yticks(positions)
    ax.set_yticklabels([FEATURE_LABELS.get(feature_cols[index], feature_cols[index]) for index in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (class probability units)")
    style_axes(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_paper_figure(fig, output / "shap_grouped_bar")

    plot_features = x.rename(columns=FEATURE_LABELS)
    # Older SHAP summary_plot uses NumPy's global RNG for point jitter. Restore
    # the caller state after making deterministic figures.
    random_state = np.random.get_state()
    accepts_rng = "rng" in inspect.signature(shap.summary_plot).parameters
    try:
        for class_idx, label in enumerate(classes):
            if accepts_rng:
                plot_rng = {"rng": np.random.default_rng(int(seed))}
            else:
                np.random.seed(int(seed))
                plot_rng = {}
            shap.summary_plot(
                values[:, :, class_idx], plot_features,
                plot_type="dot", max_display=len(feature_cols), show=False,
                plot_size=(8.5, max(4.8, 0.65 * len(feature_cols) + 1.5)),
                alpha=SCATTER_ALPHA, **plot_rng,
            )
            fig = plt.gcf()
            ax = fig.axes[0]
            ax.set_title("")
            ax.set_xlabel(f"SHAP value for Cluster {label} probability", fontsize=11)
            style_axes(ax)
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(11)
            ax.tick_params(axis="both", labelsize=11)
            for collection in ax.collections:
                if hasattr(collection, "set_sizes"):
                    collection.set_sizes([SCATTER_SIZE])
                    collection.set_alpha(SCATTER_ALPHA)
                    collection.set_linewidths([SCATTER_EDGE_WIDTH])
            # summary_plot sets font sizes explicitly; reset its colorbar too.
            for colorbar_ax in fig.axes[1:]:
                colorbar_ax.set_title("")
                colorbar_ax.xaxis.label.set_size(11)
                colorbar_ax.yaxis.label.set_size(11)
                colorbar_ax.tick_params(axis="both", labelsize=9)
            fig.tight_layout()
            save_paper_figure(fig, output / f"shap_beeswarm_cluster_{label}")
    finally:
        np.random.set_state(random_state)


@plt.rc_context(PAPER_STYLE)
def run_interpretation(selected: pd.DataFrame, feature_cols: tuple | list,
                       labels: np.ndarray, output_dir: Path, seed: int = 42) -> dict:
    """Interpret supplied labels; preserve the input row order and cluster IDs.

    Required selected columns: Filename and finite numeric feature_cols. The
    destination must be empty, so accidental reruns cannot mix output versions.
    Missing SHAP is an explicit error instead of silently omitting an analysis.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("RF/SHAP interpretation requires the 'shap' package in this Python environment") from exc

    feature_cols = list(feature_cols)
    if not feature_cols or len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols must contain unique feature names")
    required = {"Filename", *feature_cols}
    if missing := required - set(selected.columns):
        raise ValueError(f"Missing interpretation columns: {sorted(missing)}")
    if selected["Filename"].isna().any() or not selected["Filename"].is_unique:
        raise ValueError("Interpretation requires present, unique filenames")
    x = selected.loc[:, feature_cols].astype(float).reset_index(drop=True)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError("Labels must be one-dimensional and aligned to selected rows")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("K-means labels must be integer IDs")
    if len(x) < 3 or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Interpretation requires at least three rows of finite feature values")
    classes = np.unique(labels)
    if len(classes) < 2 or np.any(classes < 0) or np.any(classes >= len(CLUSTER_COLORS)):
        raise ValueError(f"Interpretation requires at least two cluster IDs in [0, {len(CLUSTER_COLORS) - 1}]")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Interpretation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    statistics = _statistics(x, labels, classes, output)
    rf = RandomForestClassifier(n_estimators=100, random_state=int(seed))
    rf.fit(x, labels)
    if not np.array_equal(rf.classes_, classes):
        raise ValueError("RF class order does not match saved K-means cluster IDs")
    prediction = rf.predict(x)
    probabilities = rf.predict_proba(x)
    accuracy = float(accuracy_score(labels, prediction))
    matrix = confusion_matrix(labels, prediction, labels=classes)
    class_names = [f"cluster_{label}" for label in classes]
    report = classification_report(labels, prediction, labels=classes,
                                   target_names=class_names, output_dict=True, zero_division=0)
    _write_json(output / "rf_training_classification_report.json", {
        "evaluation": "Resubstitution: same rows used to fit and evaluate the surrogate",
        "target": "Saved K-means cluster labels; not clinical ground truth",
        "report": report,
    })
    (output / "rf_training_classification_report.txt").write_text(
        "RF training/resubstitution fidelity to saved K-means labels\n"
        "Not a held-out or clinical generalization estimate.\n\n" +
        classification_report(labels, prediction, labels=classes, target_names=class_names,
                              digits=6, zero_division=0), encoding="utf-8")
    report_rows = [{"label": name, **row} for name, row in report.items() if isinstance(row, dict)]
    pd.DataFrame(report_rows).to_csv(output / "rf_training_classification_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix, index=pd.Index(class_names, name="kmeans_target"),
                 columns=class_names).to_csv(output / "rf_training_confusion_matrix.csv", encoding="utf-8-sig")
    identity_cols = [name for name in ("Filename", "patient_id", "Status") if name in selected.columns]
    predicted_rows = selected.loc[:, identity_cols].reset_index(drop=True).copy()
    predicted_rows["kmeans_cluster"] = labels
    predicted_rows["rf_predicted_cluster"] = prediction
    predicted_rows["agrees_with_kmeans"] = labels == prediction
    for column, label in enumerate(classes):
        predicted_rows[f"rf_probability_cluster_{label}"] = probabilities[:, column]
    predicted_rows.to_csv(output / "rf_training_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols, "rf_impurity_importance": rf.feature_importances_}).to_csv(
        output / "rf_impurity_feature_importance.csv", index=False, encoding="utf-8-sig")
    joblib.dump(rf, output / "rf_surrogate.joblib")

    # For sklearn RandomForestClassifier, raw tree output is class probability.
    # Explicit tree_path_dependent retains the legacy explainer's background
    # convention and avoids version-dependent changes to the 'auto' default.
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent", model_output="raw")
    raw_values = explainer.shap_values(x, check_additivity=True)
    values = _normalize_shap_values(raw_values, len(x), len(feature_cols), len(classes))
    expected_values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if expected_values.shape != (len(classes),) or not np.isfinite(expected_values).all():
        raise ValueError("SHAP baseline must supply one finite expected probability per RF class")
    reconstructed = expected_values[None, :] + values.sum(axis=1)
    max_error = float(np.max(np.abs(reconstructed - probabilities)))
    if not np.allclose(reconstructed, probabilities, rtol=1e-5, atol=1e-7):
        raise ValueError(f"SHAP probability additivity/class alignment failed; max absolute error={max_error}")
    mean_abs = np.abs(values).mean(axis=0)
    shap_rows = []
    for class_idx, label in enumerate(classes):
        for feature_idx, feature in enumerate(feature_cols):
            shap_rows.append({"cluster": int(label), "feature": feature,
                              "mean_abs_shap_probability": float(mean_abs[feature_idx, class_idx])})
    pd.DataFrame(shap_rows).to_csv(output / "shap_mean_abs_by_class.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols, "mean_abs_shap_across_samples_and_classes": mean_abs.mean(axis=1)}).to_csv(
        output / "shap_mean_abs_overall.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "shap_values.npz", values=values,
        axis_order=np.array(["sample", "feature", "class"]),
        filenames=selected["Filename"].astype(str).to_numpy(dtype=str),
        feature_names=np.asarray(feature_cols, dtype=str), class_labels=classes,
        raw_feature_units=np.asarray(list(_feature_units(feature_cols).values()), dtype=str),
        kmeans_labels=labels, raw_feature_values=x.to_numpy(),
        expected_probabilities=expected_values, predicted_probabilities=probabilities,
        reconstructed_probabilities=reconstructed,
    )

    render_interpretation_figures(x, labels, classes, values, matrix, accuracy, output, seed)

    repeated_patients = bool("patient_id" in selected.columns and selected["patient_id"].duplicated().any())
    metadata = {
        "n": len(x), "feature_names": feature_cols, "class_labels": classes.tolist(),
        "feature_units": _feature_units(feature_cols),
        "statistics": statistics,
        "rf": {"n_estimators": 100, "random_state": int(seed),
               "input_scale": "Raw physical diameters in their named units and dimensionless ratios; same features as K-means before z-score standardization",
               "input_feature_units": _feature_units(feature_cols),
               "target": "Saved K-means labels",
               "evaluation": "Training/resubstitution fidelity; no train-test split",
               "training_accuracy": accuracy,
               "training_macro_f1": float(report["macro avg"]["f1-score"]),
               "parameters": rf.get_params()},
        "shap": {"explainer": "TreeExplainer", "feature_perturbation": "tree_path_dependent",
                 "model_output_argument": "raw", "output_units": "Class probability for sklearn RandomForestClassifier",
                 "normalization": "[sample, feature, rf.classes_ order]", "shape": list(values.shape),
                 "expected_probabilities": expected_values.tolist(),
                 "probability_additivity_max_absolute_error": max_error,
                 "probability_additivity_rtol": 1e-5, "probability_additivity_atol": 1e-7,
                 "mean_abs_aggregation": "All included rows for each RF output class",
                 "interpretation": "Positive SHAP increases the surrogate probability of that cluster relative to its baseline"},
        "limits": [
            "Statistics reuse the clustering inputs and therefore describe derived groups rather than independently validate them.",
            "RF training fidelity measures reproduction of K-means labels and is not an estimate of clinical or held-out generalization.",
            "Mean diameter and diameter ratio are derived, correlated features; SHAP contributions depend on this feature set and the explanation convention.",
            "Thin/wide branch labels are width-ranked proxies and do not establish anatomical arterial/venous identity.",
        ] + (["Multiple capillaries share patient_id; these per-capillary Kruskal-Wallis p-values do not account for within-patient dependence."] if repeated_patients else []),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "shap": shap.__version__, "matplotlib": matplotlib.__version__},
        "plot_format": {**plot_style_metadata(),
                        "cluster_colors": {str(int(label)): CLUSTER_COLORS[int(label)] for label in classes}},
        "references": [
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
            "https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html",
        ],
    }
    _write_json(output / "interpretation_metadata.json", metadata)
    return metadata
