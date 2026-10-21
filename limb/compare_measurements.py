r"""Measure selected GT masks with five-fold U-Net MTL OOF U/D and manual lines.

This final evaluator accepts only five-fold U-Net MTL DINOv3 ViT-B/16 runs.
Output names and provenance identify the validated DINOv3 run.

분할 gt값과 수동 측정값과 pearson ,icc(c,1), icc(a,1) 계산하는 코드

The loop diameter comes from the completed model's predicted U/D coordinates.
The two branch diameters come from the current Capillary_limb analyzer applied
fresh to the GT TIFFs. Thin/estimated arterial and wide/estimated venous are
width-ranked proxies, not independently established anatomical identities.
Every finite, positive measurement is included regardless of analyzer QC
status; the status remains visible in the per-image CSV.
The output includes agreement_table_12.csv in U/Tortuous/Crossing/Total order,
each with loop, thin/estimated arterial, and wide/estimated venous rows.
Title-free scatter and Bland-Altman figures are saved for each site in µm
(default: 0.55 µm/px), as PNG and TIFF, with a combined overview PNG.
Original pixel CSV columns are retained alongside calibrated *_um columns.

PowerShell example:
    & C:\Users\USER\.conda\envs\younguk\python.exe -B C:\Users\USER\Desktop\Capillary_limb\limb\compare_measurements.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

DESKTOP = Path.home() / "Desktop"
WORKSPACE = DESKTOP / "Multi-Task-Learning"
LIMB_CHECK = DESKTOP / "Capillary_limb" / "check"
DEFAULT_RUN = (DESKTOP / "논문 결과" /
               "results_unet_mtl_dinov3_vitb16_finetune_heatmap_dbc65ae36d7e")
DEFAULT_SELECTION = DESKTOP / "Pa_capillary"
DEFAULT_SELECTION_CSV = LIMB_CHECK / "diameter_selection_141.csv"
ORIGINAL_SELECTION_CSV = LIMB_CHECK / "diameter_selection_original_144.csv"
DEFAULT_GT = DESKTOP / "GT_masks_144"
# DEFAULT_MANUAL = (DESKTOP / "Capillary_limb" / "label" / "outputs" /
#                   "section4_review_pa_141" / "manual_review_141.csv")
DEFAULT_MANUAL = (
    DESKTOP / "Capillary_limb" / "label" / "outputs"
    / "section4_review_pa_141_rater2"
    / "manual_review_141_rater2.csv"
)
DEFAULT_ANALYZER = DESKTOP / "Capillary_limb" / "limb"
ENCODER_LABELS = {"dinov3": "DINOv3 ViT-B/16"}
DEFAULT_UM_PER_PIXEL = 0.55
SHAPE_COLORS = {"u": "#2474a6", "tortuous": "#c66b20", "crossing": "#418652"}

os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / "outputs" / ".matplotlib"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(LIMB_CHECK))
import section4_oof_limb_export as oof  # noqa: E402
import section4_gt_limb_export as gt_export  # noqa: E402
import section4_validation as validation  # noqa: E402


MASTER_COLUMNS = ("filename", "fold", "pred_U_x", "pred_U_y",
                  "pred_D_x", "pred_D_y", "pred_length_px")
MEASURES = (
    ("loop", "Loop diameter", "manual_loop_px", "auto_loop_px", "predicted_ud"),
    ("thin_branch", "Thin branch (estimated arterial side)",
     "manual_thin_px", "auto_thin_px", "gt_mask"),
    ("wide_branch", "Wide branch (estimated venous side)",
     "manual_wide_px", "auto_wide_px", "gt_mask"),
)
TABLE_SHAPES = (("u", "U"), ("tortuous", "Tortuous"),
                ("crossing", "Crossing"), ("all", "Total"))
TABLE_MEASURES = (("loop", "루프"), ("thin_branch", "세동맥측"),
                  ("wide_branch", "세정맥측"))


def run_model_info(run_dir: Path) -> tuple[str, str]:
    manifest_path = run_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    spec = manifest["run_spec"]
    model = spec["experiment"]
    encoder = str(model["encoder_name"]).lower()
    if (int(spec["n_splits"]) != 5 or model["model_name"] != "unet" or
            encoder != "dinov3" or model["task_mode"] != "mtl"):
        raise ValueError("Expected a five-fold U-Net MTL run with the DINOv3 encoder")
    run_id = str(manifest.get("run_id", ""))
    if not run_id:
        raise ValueError("Run manifest has no run_id")
    return run_id, encoder


def check_run_model(run_dir: Path) -> str:
    """Retain the run-ID interface used by the predicted-mask evaluator."""
    return run_model_info(run_dir)[0]


def print_input_summary(run_dir: Path, encoder_name: str, selected_n: int,
                        selection_csv: Path, manual_csv: Path,
                        mask_dir: Path, mask_kind: str) -> None:
    """Show the resolved model/data inputs before measuring any images."""
    print(f"Model: U-Net MTL / {ENCODER_LABELS[encoder_name]}; selected={selected_n}", flush=True)
    print(f"Predicted U/D: {run_dir / 'master_predicted_coordinates.csv'}", flush=True)
    print(f"{mask_kind} masks: {mask_dir}", flush=True)
    print(f"Manual measurements: {manual_csv}", flush=True)
    print(f"Selection: {selection_csv}", flush=True)


def verify_master(run_dir: Path, prepared: list[oof.PreparedImage]) -> Path:
    """Match the complete master coordinate table to all five OOF fold tables."""
    master_path = run_dir / "master_predicted_coordinates.csv"
    master = oof.index_unique(oof.csv_rows(master_path, MASTER_COLUMNS),
                              "filename", master_path)
    encountered: set[str] = set()
    for fold in range(1, 6):
        fold_path = run_dir / f"fold_{fold}" / "predicted_coordinates.csv"
        fold_rows = oof.index_unique(oof.csv_rows(fold_path, MASTER_COLUMNS),
                                     "filename", fold_path)
        for name, row in fold_rows.items():
            if name in encountered:
                raise ValueError(f"Duplicate image across fold coordinates: {name}")
            encountered.add(name)
            if name not in master:
                raise ValueError(f"Missing master coordinate: {name}")
            main = master[name]
            if int(row["fold"]) != fold or int(main["fold"]) != fold:
                raise ValueError(f"Master/fold mismatch: {name}")
            for field in MASTER_COLUMNS[2:]:
                a, b = float(row[field]), float(main[field])
                if not (math.isfinite(a) and math.isfinite(b) and
                        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)):
                    raise ValueError(f"Master/fold {field} mismatch: {name}")
    if encountered != set(master):
        raise ValueError("Master coordinates contain images outside the OOF folds")
    for item in prepared:
        row = master[item.selected.filename]
        if int(row["fold"]) != item.fold:
            raise ValueError(f"Selected image fold mismatch: {item.selected.filename}")
        _, _, _, _, diameter = oof.finite_coordinate_values(item)
        if not math.isclose(float(row["pred_length_px"]), diameter,
                            rel_tol=1e-4, abs_tol=0.01):
            raise ValueError(f"Selected image U/D distance mismatch: {item.selected.filename}")
    return master_path


def gt_manifest_path(gt_dir: Path) -> Path:
    current = gt_dir / "gt_mask_manifest_141.csv"
    return current if current.is_file() else gt_dir / "gt_mask_manifest_144.csv"


def read_original_selection(selection_dir: Path) -> pd.DataFrame:
    """Retain the original cohort denominator after excluded images are archived."""
    active = validation.read_selected(selection_dir)
    if (selection_dir.resolve() != DEFAULT_SELECTION.resolve() or
            not ORIGINAL_SELECTION_CSV.is_file()):
        return active
    registry = oof.select_images(None, ORIGINAL_SELECTION_CSV, 144)
    original = pd.DataFrame([
        {"Filename": item.filename, "shape": item.category.lower(),
         "patient_id": item.patient_id.lower()}
        for item in registry
    ])
    if set(original["shape"]) != {"u", "tortuous", "twist"}:
        raise ValueError("Original selection registry has unexpected categories")
    reference = original.set_index("Filename")
    if not set(active["Filename"]) <= set(reference.index):
        raise ValueError("Active images must belong to the original selection registry")
    for row in active.itertuples(index=False):
        expected = reference.loc[row.Filename]
        if (row.shape != expected["shape"] or
                row.patient_id != expected["patient_id"]):
            raise ValueError(f"Active category/patient differs from original registry: {row.Filename}")
    return original.sort_values(["shape", "Filename"]).reset_index(drop=True)


def rebase_gt_manifest(gt_dir: Path, destination: Path,
                       selected: list[oof.SelectedImage] | None = None,
                       selection_dir: Path = DEFAULT_SELECTION) -> None:
    original = gt_manifest_path(gt_dir)
    rows = oof.csv_rows(original, ("Category", "Selected_Image", "GT_Original",
                                    "GT_Copy", "Original_SHA256", "Copy_SHA256", "Bytes"))
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = oof.checked_filename(Path(row["Selected_Image"]).name, str(original))
        if name in by_name:
            raise ValueError(f"Duplicate GT manifest image: {name}")
        by_name[name] = row
    if not rows:
        raise ValueError(f"GT manifest is empty: {original}")
    if selected is not None:
        chosen = {item.filename for item in selected}
        if len(chosen) != len(selected):
            raise ValueError("Duplicate selected image")
        if set(by_name) != chosen:
            original_names = set(read_original_selection(selection_dir)["Filename"])
            if (original.name != "gt_mask_manifest_144.csv" or len(by_name) != 144 or
                    set(by_name) != original_names or not chosen <= original_names):
                raise ValueError("GT manifest must match the cohort or the complete original 144 images")
        rows = [by_name[item.filename] for item in selected]
        for item, row in zip(selected, rows):
            if row["Category"] != item.category:
                raise ValueError(f"GT manifest category differs from cohort: {item.filename}")
    expected_paths: set[Path] = set()
    for row in rows:
        name = Path(row["Selected_Image"]).name
        category = row["Category"]
        if category not in {"U", "tortuous", "twist"}:
            raise ValueError(f"Unexpected GT category: {category}")
        row["GT_Copy"] = str((gt_dir / row["Category"] /
                              f"{Path(name).stem}.tif").resolve())
        expected_paths.add(Path(row["GT_Copy"]))
    actual_paths = {path.resolve()
                    for category in ("U", "tortuous", "twist")
                    for path in (gt_dir / category).iterdir()
                    if path.is_file() and path.suffix.lower() in oof.IMAGE_EXTENSIONS}
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(f"GT folder inventory differs from cohort: missing={missing}, extra={extra}")
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_selection(selection_dir: Path, manual_csv: Path,
                       selected: list[oof.SelectedImage]) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = validation.read_selected(selection_dir)
    reference = original.set_index("Filename")
    chosen = {item.filename for item in selected}
    if len(chosen) != len(selected) or not chosen <= set(reference.index):
        raise ValueError("Cohort names must be unique members of the original selection")
    for item in selected:
        row = reference.loc[item.filename]
        if (item.category.lower() != row["shape"] or
                item.patient_id.lower() != row["patient_id"]):
            raise ValueError(f"Cohort category/patient differs from original selection: {item.filename}")
    table = original.loc[original["Filename"].isin(chosen)].reset_index(drop=True)
    # Validate either the original annotation file or an exact subset against
    # the original image registry before restricting it to this cohort.
    manual = validation.read_manual(manual_csv, original)
    manual = manual.loc[manual["Filename"].isin(chosen)].reset_index(drop=True)
    if (len(manual) != len(selected) or not manual["status"].eq("approved").all() or
            set(manual["Filename"]) != chosen):
        raise ValueError("All cohort images require approved manual measurements")
    return table, manual


def write_manual_subset_exports(stage: Path, manual_csv: Path,
                                 selected: list[oof.SelectedImage]) -> dict[str, str]:
    """Export raw annotation fields without rounding or recalculating values."""
    rows = oof.csv_rows(manual_csv, ("Filename",))
    indexed = oof.index_unique(rows, "Filename", manual_csv)
    chosen = {item.filename for item in selected}
    if len(chosen) != len(selected) or not chosen <= set(indexed):
        raise ValueError("Manual subset export is missing cohort rows or contains duplicate selection")
    names_by_shape = {shape: {item.filename for item in selected
                              if item.category.lower() == category}
                       for shape, category in (("u", "u"), ("tortuous", "tortuous"),
                                               ("crossing", "twist"))}
    if set().union(*names_by_shape.values()) != chosen:
        raise ValueError("Unexpected cohort category in manual subset export")
    exports: dict[str, str] = {}
    for suffix, names in [("", chosen), *[(f"_{shape}", members)
                                          for shape, members in names_by_shape.items()]]:
        path = stage / f"manual_review_{len(selected)}{suffix}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(row for row in rows if row["Filename"] in names)
        exports[path.name] = oof.file_sha256(path)
    return exports


def build_comparison(selected: pd.DataFrame, manual: pd.DataFrame,
                     measurements: list[dict]) -> pd.DataFrame:
    rows = []
    for record in measurements:
        b1 = pd.to_numeric(record["Branch1_Diameter_px"], errors="coerce")
        b2 = pd.to_numeric(record["Branch2_Diameter_px"], errors="coerce")
        measured = bool(np.isfinite(b1) and np.isfinite(b2) and b1 > 0 and b2 > 0)
        rows.append({
            "Filename": record["Filename"], "fold": int(record["fold"]),
            "run_id": record["Run_ID"], "gt_mask": record["GT_Mask"],
            "gt_mask_sha256": record["GT_Mask_SHA256"],
            "algorithm_sha256": record["Algorithm_SHA256"],
            "gt_status": record["Status"], "gt_reason": record["Reason"],
            "auto_loop_px": float(record["Loop_Diameter_px"]),
            "gt_branch1_px": b1, "gt_branch2_px": b2,
            "auto_thin_px": min(b1, b2) if measured else np.nan,
            "auto_wide_px": max(b1, b2) if measured else np.nan,
            "gt_measured_pair": measured,
        })
    measured_table = pd.DataFrame(rows)
    table = selected[["Filename", "shape", "patient_id"]].merge(
        manual, on="Filename", how="left", validate="one_to_one")
    table = table.merge(measured_table, on="Filename", how="left",
                        validate="one_to_one", indicator=True)
    if (len(table) != len(selected) or not table["_merge"].eq("both").all() or
            not table["Filename"].is_unique):
        raise ValueError("GT measurements do not match the selected cohort")
    table = table.drop(columns="_merge")
    if not table["gt_status"].isin(["Success", "Uncertain", "Failed"]).all():
        raise ValueError("Unexpected analyzer QC status")
    for _, _, manual_col, auto_col, _ in MEASURES:
        table[auto_col.replace("auto_", "error_")] = table[auto_col] - table[manual_col]
    table["shape_report"] = table["shape"].replace({"twist": "crossing"})
    return table


def summarize(table: pd.DataFrame, shape: str = "all") -> pd.DataFrame:
    subset = table if shape == "all" else table.loc[table["shape_report"] == shape]
    result = []
    status_counts = subset["gt_status"].value_counts()
    for measure, label, manual_col, auto_col, source in MEASURES:
        manual = pd.to_numeric(subset[manual_col], errors="coerce")
        auto = pd.to_numeric(subset[auto_col], errors="coerce")
        eligible = (subset["status"].eq("approved") & np.isfinite(manual) &
                    np.isfinite(auto) & (manual > 0) & (auto > 0))
        checked = subset.loc[eligible]
        if len(checked) < 3:
            raise ValueError(f"Too few valid {shape}/{measure} measurements")
        values = validation._point_metrics(
            manual.loc[eligible].to_numpy(dtype=float),
            auto.loc[eligible].to_numpy(dtype=float))
        result.append({
            "shape": shape, "measure": measure, "label": label,
            "source": source, "n": len(checked), "selected_n": len(subset),
            "patient_n": int(checked["patient_id"].nunique()),
            "success_n": int(status_counts.get("Success", 0)),
            "uncertain_n": int(status_counts.get("Uncertain", 0)),
            "failed_n": int(status_counts.get("Failed", 0)),
            "inclusion_rule": "approved + finite positive measured values, any QC status",
            **values,
        })
    return pd.DataFrame(result)


def make_agreement_table_12(summary: pd.DataFrame,
                            by_shape: pd.DataFrame) -> pd.DataFrame:
    """Arrange the 3 shapes and total into the requested thesis table."""
    combined = pd.concat([by_shape, summary], ignore_index=True)
    expected = {(shape, measure) for shape, _ in TABLE_SHAPES
                for measure, _ in TABLE_MEASURES}
    observed = list(zip(combined["shape"], combined["measure"]))
    if len(observed) != 12 or set(observed) != expected or len(set(observed)) != 12:
        raise ValueError("Expected exactly one result for each of the 12 table rows")
    indexed = combined.set_index(["shape", "measure"])
    rows = []
    for shape, shape_label in TABLE_SHAPES:
        for measure, measure_label in TABLE_MEASURES:
            result = indexed.loc[(shape, measure)]
            rows.append({
                "형태": shape_label,
                "측정 부위": measure_label,
                "Pearson r": float(result["pearson_r"]),
                "ICC(C,1)": float(result["icc_consistency_C1"]),
                "ICC(A,1)": float(result["icc_absolute_A1"]),
            })
    table = pd.DataFrame(rows)
    if not np.isfinite(table[["Pearson r", "ICC(C,1)", "ICC(A,1)"]]
                       .to_numpy(dtype=float)).all():
        raise ValueError("Non-finite value in the 12-row agreement table")
    return table


def add_micrometer_columns(table: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame:
    """Keep raw pixels and add explicitly named calibrated measurements."""
    if not np.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError("Micrometers per pixel must be finite and positive")
    converted = {column.replace("_px", "_um"):
                 table[column].to_numpy(dtype=float) * um_per_pixel
                 for column in table
                 if column.endswith("_px") or "_px_ci95_" in column}
    return pd.concat([table, pd.DataFrame(converted, index=table.index)], axis=1)


def figure_pairs(table: pd.DataFrame, manual_col: str, auto_col: str,
                 summary: pd.Series) -> pd.DataFrame:
    """Use exactly the same eligible observations as the agreement statistics."""
    manual = pd.to_numeric(table[manual_col], errors="coerce")
    auto = pd.to_numeric(table[auto_col], errors="coerce")
    eligible = (table["status"].eq("approved") & np.isfinite(manual) &
                np.isfinite(auto) & (manual > 0) & (auto > 0))
    pairs = table.loc[eligible]
    if len(pairs) != int(summary["n"]):
        raise ValueError("Figure and agreement summary have different sample counts")
    if not pairs["shape_report"].isin(SHAPE_COLORS).all():
        raise ValueError("Unexpected shape in figure data")
    return pairs


def draw_scatter_panel(ax, pairs: pd.DataFrame, manual_col: str,
                       auto_col: str) -> None:
    manual, auto = pairs[manual_col.replace("_px", "_um")], pairs[auto_col.replace("_px", "_um")]
    for shape, label in TABLE_SHAPES[:-1]:
        part = pairs["shape_report"].eq(shape)
        ax.scatter(manual[part], auto[part], s=22, alpha=0.75,
                   color=SHAPE_COLORS[shape], label=label)
    lower, upper = min(manual.min(), auto.min()), max(manual.max(), auto.max())
    margin = max((upper - lower) * 0.06, 0.5)
    limits = (lower - margin, upper + margin)
    ax.plot(limits, limits, "--", color="#666666", linewidth=1.2)
    ax.set(xlim=limits, ylim=limits,
           xlabel="Manual diameter (µm)", ylabel="Automatic diameter (µm)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)


def draw_bland_altman_panel(ax, pairs: pd.DataFrame, summary: pd.Series,
                           manual_col: str, auto_col: str) -> None:
    manual, auto = pairs[manual_col.replace("_px", "_um")], pairs[auto_col.replace("_px", "_um")]
    mean, difference = (auto + manual) / 2, auto - manual
    bias, sd = difference.mean(), difference.std(ddof=1)
    if not np.allclose([bias, bias - 1.96 * sd, bias + 1.96 * sd],
                       summary[["bias_um", "loa_lower_um", "loa_upper_um"]].to_numpy(dtype=float),
                       atol=1e-10, rtol=0):
        raise ValueError("Plot data and micrometer summary disagree")
    for shape, _ in TABLE_SHAPES[:-1]:
        part = pairs["shape_report"].eq(shape)
        ax.scatter(mean[part], difference[part], s=22, alpha=0.75,
                   color=SHAPE_COLORS[shape])
    for key, label, style in (("bias_um", "Mean", "-"),
                               ("loa_lower_um", "Lower 95% LoA", "--"),
                               ("loa_upper_um", "Upper 95% LoA", "--")):
        value = float(summary[key])
        ax.axhline(value, color="#944444", linestyle=style, linewidth=1.2,
                   label=f"{label}: {value:.2f} µm")
    ax.set(xlabel="Mean of manual and automatic (µm)",
           ylabel="Automatic − Manual (µm)")
    ax.legend(loc="best", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)


def save_figure(fig, stem: Path, tiff: bool = True) -> list[str]:
    """Save preview PNG and losslessly compressed publication TIFF."""
    files = []
    for extension, dpi in (("png", 300), ("tiff", 600)):
        if extension == "tiff" and not tiff:
            continue
        path = stem.with_suffix(f".{extension}")
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        fig.savefig(path, dpi=dpi, facecolor="white", **options)
        files.append(path.name)
    plt.close(fig)
    return files


def draw_figures(table: pd.DataFrame, summary: pd.DataFrame,
                 output: Path) -> list[str]:
    """Title-free per-site scatter/BA plots, paired plots, and a 3-by-2 overview."""
    files = []
    style = {"font.family": "sans-serif", "font.size": 11,
             "axes.titlesize": 11, "axes.labelsize": 11}
    with plt.rc_context(style):
        overview, overview_axes = plt.subplots(3, 2, figsize=(11.5, 13.0), constrained_layout=True)
        for index, (measure, _, manual_col, auto_col, _) in enumerate(MEASURES):
            row = summary.loc[summary["measure"].eq(measure)].iloc[0]
            pairs = figure_pairs(table, manual_col, auto_col, row)
            draw_scatter_panel(overview_axes[index, 0], pairs, manual_col, auto_col)
            draw_bland_altman_panel(overview_axes[index, 1], pairs, row, manual_col, auto_col)
            pair, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
            draw_scatter_panel(axes[0], pairs, manual_col, auto_col)
            draw_bland_altman_panel(axes[1], pairs, row, manual_col, auto_col)
            files.extend(save_figure(pair, output / f"{measure}_scatter_bland_altman_um"))
            scatter, ax = plt.subplots(figsize=(5.4, 5.1), constrained_layout=True)
            draw_scatter_panel(ax, pairs, manual_col, auto_col)
            files.extend(save_figure(scatter, output / f"{measure}_scatter_um"))
            bland, ax = plt.subplots(figsize=(5.8, 5.1), constrained_layout=True)
            draw_bland_altman_panel(ax, pairs, row, manual_col, auto_col)
            files.extend(save_figure(bland, output / f"{measure}_bland_altman_um"))
        files.extend(save_figure(overview, output / "scatter_bland_altman_um", tiff=False))
    return files


def figure_provenance(um_per_pixel: float, files: list[str]) -> dict:
    return {
        "calibration": {"um_per_pixel": um_per_pixel, "source": "User-specified calibration",
                        "raw_unit": "px", "figure_unit": "µm",
                        "conversion": "Pixel length and error columns retained; matching *_um columns multiplied by um_per_pixel"},
        "difference_direction": "Automatic minus manual",
        "bland_altman": {"mean": "(Automatic + manual) / 2", "unit": "µm",
                          "limits": "mean difference +/- 1.96 * sample SD of differences (ddof=1)",
                          "figures": files, "titles": False, "statistic_heading": False,
                          "png_dpi": 300, "tiff_dpi": 600},
    }


def execute(args: argparse.Namespace) -> None:
    print(f"Evaluation script: {Path(__file__).resolve()}", flush=True)
    run_dir = args.run_dir.resolve()
    expected_run_id, encoder_name = run_model_info(run_dir)
    selection_dir = args.selection_dir.resolve()
    selection_csv = args.selection_csv.resolve()
    selection_sha256 = oof.file_sha256(selection_csv)
    selected = oof.select_images(None, selection_csv, args.expected_count)
    selected_n = len(selected)
    original_table = read_original_selection(selection_dir)
    if args.output_dir is None:
        default_output = (WORKSPACE / "outputs" /
                          f"gt_measurements_{encoder_name}_{selected_n}_{datetime.now():%Y%m%d}")
        output = default_output.resolve()
        counter = 2
        while output.exists():
            output = default_output.with_name(
                f"{default_output.name}_run{counter}").resolve()
            counter += 1
    else:
        output = args.output_dir.resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists; choose a fresh --output-dir: {output}")
    gt_dir = args.gt_dir.resolve()
    print(f"Planned output directory: {output}", flush=True)
    prepared, run_id = oof.verify_run(run_dir, selected)
    if run_id != expected_run_id or len({item.fold for item in prepared}) != 5:
        raise ValueError("Run ID or selected fold coverage mismatch")
    master_path = verify_master(run_dir, prepared)
    manual_csv = args.manual_csv.resolve()
    manual_sha256 = oof.file_sha256(manual_csv)
    selection_table, manual = validate_selection(selection_dir,
                                                  manual_csv, selected)
    print_input_summary(run_dir, encoder_name, selected_n, selection_csv,
                        manual_csv, gt_dir, "GT")
    with tempfile.TemporaryDirectory(prefix=f"gt_{selected_n}_manifest_") as temp:
        rebased = Path(temp) / "gt_mask_manifest_rebased.csv"
        rebase_gt_manifest(gt_dir, rebased, selected, selection_dir)
        matches = gt_export.verify_gt_matches(selected, prepared,
                                              selection_dir, gt_dir, rebased)
        if len(matches) != selected_n:
            raise ValueError("GT verification did not return the expected cohort masks")
        print(f"Verified {len(matches)} GT masks and five OOF folds; run={run_id}",
              flush=True)
        if getattr(args, "dry_run", False):
            print("Input check passed; no measurements or output files were written.", flush=True)
            return
        measurements = gt_export.export_gt_rows(matches, run_id,
                                                 args.analyzer_dir.resolve())
        comparison = build_comparison(selection_table, manual, measurements)
        summary = summarize(comparison)
        by_shape = pd.concat([summarize(comparison, name)
                              for name in ("u", "tortuous", "crossing")],
                             ignore_index=True)
        table_12 = make_agreement_table_12(summary, by_shape)
        comparison = add_micrometer_columns(comparison, args.um_per_pixel)
        summary = add_micrometer_columns(summary, args.um_per_pixel)
        by_shape = add_micrometer_columns(by_shape, args.um_per_pixel)
        if not np.isfinite(summary[["pearson_r", "icc_absolute_A1",
                                    "icc_consistency_C1", "bias_px",
                                    "loa_lower_px", "loa_upper_px",
                                    "mae_px"]].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite summary statistic")
        rebased_bytes = rebased.read_bytes()
    if oof.file_sha256(manual_csv) != manual_sha256:
        raise RuntimeError("Manual CSV changed during analysis; rerun with stable annotations")
    if oof.file_sha256(selection_csv) != selection_sha256:
        raise RuntimeError("Cohort CSV changed during analysis; rerun with stable selection")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}_stage_",
                                     dir=output.parent) as temporary_output:
        stage = Path(temporary_output)
        comparison.to_csv(stage / "comparison_by_image.csv", index=False,
                          encoding="utf-8-sig")
        summary.to_csv(stage / "agreement_summary.csv", index=False,
                       encoding="utf-8-sig")
        by_shape.to_csv(stage / "agreement_by_shape.csv", index=False,
                        encoding="utf-8-sig")
        table_12.to_csv(stage / "agreement_table_12.csv", index=False,
                        encoding="utf-8-sig")
        with (stage / f"gt_measurements_{selected_n}.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=gt_export.FIELDNAMES)
            writer.writeheader()
            writer.writerows(measurements)
        (stage / "gt_mask_manifest_rebased.csv").write_bytes(rebased_bytes)
        shutil.copyfile(selection_csv, stage / f"diameter_selection_{selected_n}.csv")
        manual_subset_files = write_manual_subset_exports(stage, manual_csv, selected)
        figure_files = draw_figures(comparison, summary, stage)
        metadata = {
            **figure_provenance(args.um_per_pixel, figure_files),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "design": f"Patient-held-out 5-fold U-Net MTL with {ENCODER_LABELS[encoder_name]} encoder",
            "encoder_name": encoder_name,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "master_predicted_coordinates": str(master_path),
            "master_predicted_coordinates_sha256": oof.file_sha256(master_path),
            "selection_dir": str(selection_dir),
            "selection_source": "Explicit cohort CSV, verified against original image registry",
            "selection_csv": str(selection_csv),
            "selection_csv_sha256": selection_sha256,
            "original_n": len(original_table),
            "excluded_n": len(original_table) - selected_n,
            "excluded_filenames": sorted(set(original_table["Filename"]) -
                                         {item.filename for item in selected}),
            "manual_csv": str(manual_csv),
            "manual_csv_sha256": manual_sha256,
            "manual_subset_exports_sha256": manual_subset_files,
            "gt_mask_dir": str(gt_dir),
            "original_gt_manifest": str(gt_manifest_path(gt_dir)),
            "original_gt_manifest_sha256": oof.file_sha256(
                gt_manifest_path(gt_dir)),
            "analyzer_dir": str(args.analyzer_dir.resolve()),
            "algorithm_sha256": measurements[0]["Algorithm_SHA256"],
            "selected_n": len(comparison),
            "gt_qc_counts": {str(k): int(v) for k, v in
                             comparison["gt_status"].value_counts().items()},
            "measured_pair_n": int(comparison["gt_measured_pair"].sum()),
            "branch_identity": "Thin/estimated arterial and wide/estimated venous are width-ranked proxies, not independently confirmed anatomical identities.",
            "inclusion_rule": "All approved images with finite positive measurements, including analyzer QC Uncertain",
            "loop_source": "Predicted U/D distance from master coordinates, verified against corresponding outer-test fold coordinates",
            "branch_source": "Current limb analyzer run on GT segmentation TIFFs using the same predicted U/D coordinates",
            "agreement_table_12_rows": len(table_12),
            "agreement_table_12_branch_labels": "세동맥측/세정맥측 are width-ranked thin/wide proxies, not confirmed anatomical identities",
        }
        (stage / "provenance.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared during analysis: {output}")
        if oof.file_sha256(manual_csv) != manual_sha256:
            raise RuntimeError("Manual CSV changed while writing results; rerun")
        if oof.file_sha256(selection_csv) != selection_sha256:
            raise RuntimeError("Cohort CSV changed while writing results; rerun")
        # TemporaryDirectory has owner-only ACLs on Windows. A fresh copy
        # inherits normal workspace permissions so the user can open results.
        shutil.copytree(stage, output)
    print(f"Saved GT-only comparison to {output}", flush=True)
    print(f"Saved {len(figure_files)} figures in µm (1 px = {args.um_per_pixel:g} µm); difference = automatic - manual.", flush=True)
    console_table = table_12.rename(columns={"형태": "Shape", "측정 부위": "Measure"})
    console_table["Measure"] = console_table["Measure"].replace({
        "루프": "Loop", "세동맥측": "Thin branch", "세정맥측": "Wide branch",
    })
    print(console_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV,
                        help="Explicit cohort membership CSV (Filename and Category)")
    parser.add_argument("--expected-count", type=int, default=141)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--analyzer-dir", type=Path, default=DEFAULT_ANALYZER)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate model, paths, cohort, manual values and masks without writing results")
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh output directory (default: next unused numbered run)")
    parser.add_argument("--um-per-pixel", type=float, default=DEFAULT_UM_PER_PIXEL,
                        help="Micrometers per pixel for figures and *_um CSV columns (default: 0.55)")
    args = parser.parse_args()
    if not np.isfinite(args.um_per_pixel) or args.um_per_pixel <= 0:
        parser.error("--um-per-pixel must be a finite positive number")
    execute(args)


if __name__ == "__main__":
    main()
