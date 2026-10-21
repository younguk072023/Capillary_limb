"""Audit Section 4 manual labels and patient-held-out MTL measurements.

This script does not run a model or change source labels. It joins one selected
ROI to exactly one outer-test fold, preserves failed algorithm cases, and only
calculates agreement on manually approved, valid measurement pairs.

Example (PowerShell)::

    python check/section4_validation.py --selection-dir C:/Users/USER/Desktop/analysis/image \
        --oof-root results_unet_mtl_heatmap_bce_dice_5849db92da83 \
        --manual-csv label/outputs/section4_review/review.csv \
        --limb-csv check/section4_oof_limb.csv \
        --output-dir check/section4_validation_output

The manual CSV is created by label/section4_manual_review.py. Until it exists,
omit --manual-csv and --limb-csv to audit only the 51-image OOF manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SHAPES = ("u", "tortuous", "twist")
POINT_COLUMNS = (
    "B1_x1", "B1_y1", "B1_x2", "B1_y2",
    "B2_x1", "B2_y1", "B2_x2", "B2_y2",
    "U_x", "U_y", "D_x", "D_y",
)
MEASURE_COLUMNS = ("manual_loop_px", "manual_thin_px", "manual_wide_px")


def require_columns(table: pd.DataFrame, columns: tuple[str, ...], source: Path) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def require_unique(table: pd.DataFrame, key: str, source: Path) -> None:
    repeated = table.loc[table[key].duplicated(False), key].astype(str).unique()
    if len(repeated):
        raise ValueError(f"{source}: duplicate {key}: {repeated[:10].tolist()}")


def read_selected(selection_dir: Path) -> pd.DataFrame:
    rows = []
    for shape in SHAPES:
        folder = selection_dir / shape
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing shape folder: {folder}")
        for path in sorted(folder.glob("*.png")):
            patient = re.match(r"^(p\d+)_", path.name, re.IGNORECASE)
            if not patient:
                raise ValueError(f"Cannot extract patient ID: {path.name}")
            rows.append({"Filename": path.name, "shape": shape,
                         "patient_id": patient.group(1).lower(),
                         "original_image": str(path.resolve())})
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError(f"No selected PNGs: {selection_dir}")
    require_unique(selected, "Filename", selection_dir)
    return selected.sort_values(["shape", "Filename"]).reset_index(drop=True)


def read_oof(oof_root: Path, selected: pd.DataFrame) -> pd.DataFrame:
    fold_dirs = sorted((p for p in oof_root.glob("fold_*") if p.is_dir()),
                       key=lambda p: int(p.name.split("_")[-1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories: {oof_root}")
    chosen = set(selected["Filename"])
    expected_patient = selected.set_index("Filename")["patient_id"].to_dict()
    rows = []
    for folder in fold_dirs:
        fold = int(folder.name.split("_")[-1])
        split_file = folder / "patient_split.csv"
        coords_file = folder / "predicted_coordinates.csv"
        if not split_file.exists() or not coords_file.exists():
            raise FileNotFoundError(f"Missing split or coordinates in {folder}")
        split = pd.read_csv(split_file)
        coords = pd.read_csv(coords_file)
        require_columns(split, ("filename", "patient_id", "split"), split_file)
        require_columns(coords, ("filename", "fold", "pred_U_x", "pred_U_y",
                                 "pred_D_x", "pred_D_y", "pred_length_px"), coords_file)
        require_unique(split, "filename", split_file)
        require_unique(coords, "filename", coords_file)
        test = split.loc[split["split"] == "outer_test"]
        test_patients = set(test["patient_id"])
        if test_patients & set(split.loc[split["split"] != "outer_test", "patient_id"]):
            raise ValueError(f"Patient leakage in {split_file}")
        test = test.loc[test["filename"].isin(chosen)]
        joined = test.merge(coords, on="filename", how="left", validate="one_to_one",
                            indicator=True)
        if not (joined["_merge"] == "both").all():
            missing = joined.loc[joined["_merge"] != "both", "filename"].tolist()
            raise ValueError(f"Missing OOF coordinates in {folder}: {missing}")
        for _, row in joined.iterrows():
            filename = str(row["filename"])
            if str(row["patient_id"]).lower() != expected_patient[filename]:
                raise ValueError(f"Patient ID mismatch: {filename}")
            if int(row["fold"]) != fold:
                raise ValueError(f"Coordinate fold mismatch: {filename}")
            mask = folder / "predicted_masks" / f"{Path(filename).stem}.png"
            if not mask.is_file():
                raise FileNotFoundError(f"Missing predicted mask: {mask}")
            uv = [row[c] for c in ("pred_U_x", "pred_U_y", "pred_D_x", "pred_D_y")]
            finite_uv = all(pd.notna(v) and np.isfinite(float(v)) for v in uv)
            pred_loop = float(row["pred_length_px"]) if pd.notna(row["pred_length_px"]) else np.nan
            if finite_uv:
                recomputed = math.dist((float(uv[0]), float(uv[1])),
                                       (float(uv[2]), float(uv[3])))
                if not np.isfinite(pred_loop) or abs(pred_loop - recomputed) > 0.02:
                    raise ValueError(f"Inconsistent predicted loop diameter: {filename}")
            elif np.isfinite(pred_loop):
                raise ValueError(f"Loop diameter exists without four points: {filename}")
            rows.append({"Filename": filename, "oof_fold": fold,
                         "pred_loop_px": pred_loop,
                         "pred_U_x": uv[0], "pred_U_y": uv[1],
                         "pred_D_x": uv[2], "pred_D_y": uv[3],
                         "predicted_mask": str(mask.resolve())})
    oof = pd.DataFrame(rows)
    require_unique(oof, "Filename", oof_root)
    missed = chosen - set(oof["Filename"])
    if missed:
        raise ValueError(f"Selected ROIs without outer-test predictions: {sorted(missed)}")
    return oof


def _positive_distance(row: pd.Series, names: tuple[str, str, str, str]) -> float:
    values = [float(row[name]) for name in names]
    if not all(np.isfinite(values)):
        raise ValueError(f"Non-finite endpoints: {row['Filename']}, {names}")
    distance = math.dist(values[:2], values[2:])
    if distance <= 0:
        raise ValueError(f"Zero-length manual line: {row['Filename']}, {names}")
    return distance


def read_manual(path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(table, ("Filename", "status", *POINT_COLUMNS,
                            *MEASURE_COLUMNS), path)
    require_unique(table, "Filename", path)
    unknown = set(table["Filename"]) - set(selected["Filename"])
    if unknown:
        raise ValueError(f"Manual CSV has unselected images: {sorted(unknown)[:10]}")
    table["status"] = table["status"].fillna("pending").astype(str).str.lower().str.strip()
    bad_status = set(table["status"]) - {"pending", "approved", "unmeasurable"}
    if bad_status:
        raise ValueError(f"Unknown manual status: {sorted(bad_status)}")
    if "reason" in table.columns:
        without_reason = table.loc[(table["status"] == "unmeasurable") &
                                   table["reason"].fillna("").astype(str).str.strip().eq(""),
                                   "Filename"].tolist()
        if without_reason:
            raise ValueError(f"Unmeasurable reviews need reasons: {without_reason}")
    if "patient_id" in table.columns or "shape" in table.columns:
        expected = selected.set_index("Filename")
        for _, row in table.iterrows():
            name = row["Filename"]
            for key in ("patient_id", "shape"):
                if key in table.columns and str(row[key]).strip().lower() != expected.at[name, key]:
                    raise ValueError(f"Manual {key} mismatch: {name}")
    table["manual_branch1_px"] = np.nan
    table["manual_branch2_px"] = np.nan
    approved = table["status"] == "approved"
    for index, row in table.loc[approved].iterrows():
        b1 = _positive_distance(row, ("B1_x1", "B1_y1", "B1_x2", "B1_y2"))
        b2 = _positive_distance(row, ("B2_x1", "B2_y1", "B2_x2", "B2_y2"))
        loop = _positive_distance(row, ("U_x", "U_y", "D_x", "D_y"))
        for key, expected in (("manual_loop_px", loop),
                              ("manual_thin_px", min(b1, b2)),
                              ("manual_wide_px", max(b1, b2))):
            actual = float(row[key])
            if not np.isfinite(actual) or abs(actual - expected) > 0.02:
                raise ValueError(f"Stale manual diameter {key}: {row['Filename']}")
        table.at[index, "manual_branch1_px"] = b1
        table.at[index, "manual_branch2_px"] = b2
    return table[["Filename", "status", *MEASURE_COLUMNS,
                  "manual_branch1_px", "manual_branch2_px"]]


def read_limb(path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(table, ("Filename", "fold", "Status", "Reason",
                            "Branch1_Diameter_px", "Branch2_Diameter_px",
                            "Run_ID", "Predicted_Mask", "Mask_SHA256",
                            "Algorithm_SHA256"), path)
    require_unique(table, "Filename", path)
    unknown = set(table["Filename"]) - set(selected["Filename"])
    missing = set(selected["Filename"]) - set(table["Filename"])
    if unknown or missing:
        raise ValueError(f"Limb CSV does not match selection: unknown={sorted(unknown)}, "
                         f"missing={sorted(missing)}")
    table = table.rename(columns={"fold": "limb_fold", "Status": "limb_status",
                                  "Reason": "limb_reason"})
    for _, row in table.loc[table["limb_status"] == "Success"].iterrows():
        b1 = float(row["Branch1_Diameter_px"])
        b2 = float(row["Branch2_Diameter_px"])
        if not all(np.isfinite((b1, b2))) or min(b1, b2) <= 0:
            raise ValueError(f"Invalid successful limb widths: {row['Filename']}")
    table["pred_thin_px"] = table[["Branch1_Diameter_px", "Branch2_Diameter_px"]].min(axis=1, skipna=False)
    table["pred_wide_px"] = table[["Branch1_Diameter_px", "Branch2_Diameter_px"]].max(axis=1, skipna=False)
    table.loc[table["limb_status"] != "Success", ["pred_thin_px", "pred_wide_px"]] = np.nan
    return table[["Filename", "limb_fold", "limb_status", "limb_reason",
                  "pred_thin_px", "pred_wide_px",
                  "Branch1_Diameter_px", "Branch2_Diameter_px",
                  "Run_ID", "Predicted_Mask", "Mask_SHA256", "Algorithm_SHA256"]]


def icc_pair(manual: np.ndarray, automated: np.ndarray) -> tuple[float, float]:
    """Single-measure two-way absolute agreement ICC(A,1) and consistency ICC(C,1)."""
    x = np.column_stack((manual, automated)).astype(float)
    n, k = x.shape
    if n < 3:
        return np.nan, np.nan
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = x - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator_c = ms_rows + (k - 1) * ms_error
    denominator_a = denominator_c + k * (ms_cols - ms_error) / n
    agreement = (ms_rows - ms_error) / denominator_a if denominator_a else np.nan
    consistency = (ms_rows - ms_error) / denominator_c if denominator_c else np.nan
    return float(agreement), float(consistency)


def _point_metrics(manual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    diff = pred - manual
    sd = float(np.std(diff, ddof=1))
    agreement, consistency = icc_pair(manual, pred)
    return {
        "pearson_r": float(np.corrcoef(manual, pred)[0, 1])
        if np.ptp(manual) and np.ptp(pred) else np.nan,
        "icc_absolute_A1": agreement,
        "icc_consistency_C1": consistency,
        "bias_px": float(np.mean(diff)),
        "loa_lower_px": float(np.mean(diff) - 1.96 * sd),
        "loa_upper_px": float(np.mean(diff) + 1.96 * sd),
        "mae_px": float(np.mean(np.abs(diff))),
    }


def _patient_bootstrap(valid: pd.DataFrame, manual: np.ndarray, pred: np.ndarray,
                       repetitions: int) -> dict[str, float]:
    """Percentile confidence intervals by resampling patients, not individual ROIs."""
    if repetitions <= 0 or valid["patient_id"].nunique() < 3:
        return {}
    patients = valid["patient_id"].to_numpy()
    unique_patients = np.unique(patients)
    positions = {patient: np.flatnonzero(patients == patient)
                 for patient in unique_patients}
    rng = np.random.default_rng(20260928)
    draws: dict[str, list[float]] = {}
    for _ in range(repetitions):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([positions[patient] for patient in sampled])
        if len(indices) < 3:
            continue
        for key, value in _point_metrics(manual[indices], pred[indices]).items():
            if np.isfinite(value):
                draws.setdefault(key, []).append(value)
    intervals = {}
    for key, values in draws.items():
        if len(values) >= max(50, repetitions // 2):
            intervals[f"{key}_ci95_low"] = float(np.percentile(values, 2.5))
            intervals[f"{key}_ci95_high"] = float(np.percentile(values, 97.5))
    return intervals


def summary_row(table: pd.DataFrame, target: str, shape: str,
                manual_column: str, pred_column: str,
                bootstrap_reps: int = 1000) -> dict:
    valid = table.loc[(table["status"] == "approved") &
                      np.isfinite(pd.to_numeric(table[manual_column], errors="coerce")) &
                      np.isfinite(pd.to_numeric(table[pred_column], errors="coerce"))]
    manual = valid[manual_column].to_numpy(dtype=float)
    pred = valid[pred_column].to_numpy(dtype=float)
    result = {"target": target, "shape": shape,
              "selected_n": len(table),
              "approved_n": int((table["status"] == "approved").sum()),
              "paired_n": len(valid), "patient_n": valid["patient_id"].nunique()}
    if len(valid) < 3:
        return result
    result.update(_point_metrics(manual, pred))
    result.update(_patient_bootstrap(valid, manual, pred, bootstrap_reps))
    return result


def validate(selection_dir: Path, oof_root: Path, manual_csv: Path | None,
             limb_csv: Path | None, bootstrap_reps: int = 1000
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = read_selected(selection_dir)
    oof = read_oof(oof_root, selected)
    table = selected.merge(oof, on="Filename", how="left", validate="one_to_one")
    if manual_csv is not None:
        manual = read_manual(manual_csv, selected)
        table = table.merge(manual, on="Filename", how="left", validate="one_to_one")
        table["status"] = table["status"].fillna("pending")
    else:
        table["status"] = "pending"
    if limb_csv is not None:
        limb = read_limb(limb_csv, selected)
        table = table.merge(limb, on="Filename", how="left", validate="one_to_one")
        if not (table["limb_fold"] == table["oof_fold"]).all():
            bad = table.loc[table["limb_fold"] != table["oof_fold"], "Filename"].tolist()
            raise ValueError(f"Limb prediction is not from the selected ROI's outer-test fold: {bad}")
        manifest = json.loads((oof_root / "experiment_manifest.json").read_text(encoding="utf-8-sig"))
        run_id = str(manifest.get("run_id", ""))
        if not run_id or not table["Run_ID"].astype(str).eq(run_id).all():
            raise ValueError("Limb CSV Run_ID does not match the MTL run")
        if table["Algorithm_SHA256"].isna().any() or table["Algorithm_SHA256"].nunique() != 1:
            raise ValueError("Limb CSV lacks one consistent algorithm source hash")
        for _, row in table.iterrows():
            mask = Path(row["predicted_mask"]).resolve()
            if Path(row["Predicted_Mask"]).resolve() != mask:
                raise ValueError(f"Limb CSV mask path does not match OOF mask: {row['Filename']}")
            actual_hash = hashlib.sha256(mask.read_bytes()).hexdigest()
            if str(row["Mask_SHA256"]) != actual_hash:
                raise ValueError(f"Limb CSV mask content changed: {row['Filename']}")
    summary = []
    for shape in ("all", *SHAPES):
        part = table if shape == "all" else table.loc[table["shape"] == shape]
        if manual_csv is not None:
            summary.append(summary_row(part, "loop", shape, "manual_loop_px",
                                       "pred_loop_px", bootstrap_reps))
            if limb_csv is not None:
                summary.append(summary_row(part, "estimated_arterial_thin", shape,
                                           "manual_thin_px", "pred_thin_px", bootstrap_reps))
                summary.append(summary_row(part, "estimated_venous_wide", shape,
                                           "manual_wide_px", "pred_wide_px", bootstrap_reps))
    return table, pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection-dir", type=Path, required=True,
                        help="Folder containing u/tortuous/twist PNG subfolders")
    parser.add_argument("--oof-root", type=Path, required=True,
                        help="Completed MTL run with fold_*/patient_split.csv and predicted_coordinates.csv")
    parser.add_argument("--manual-csv", type=Path, help="Approved Section 4 manual review CSV")
    parser.add_argument("--limb-csv", type=Path, help="Section 4 OOF limb exporter CSV with fold provenance")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1000,
                        help="Patient-cluster bootstrap replicates for 95%% CIs (default: 1000; 0 disables)")
    args = parser.parse_args()
    if args.limb_csv and not args.manual_csv:
        parser.error("--limb-csv requires --manual-csv")
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be nonnegative")
    table, metrics = validate(args.selection_dir, args.oof_root,
                              args.manual_csv, args.limb_csv, args.bootstrap_reps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "section4_pairs_and_coverage.csv", index=False)
    metrics.to_csv(args.output_dir / "section4_agreement.csv", index=False)
    counts = table["status"].value_counts().to_dict()
    print(f"Selected: {len(table)} ROIs, {table['patient_id'].nunique()} patients")
    print(f"OOF folds: {table['oof_fold'].value_counts().sort_index().to_dict()}")
    print(f"Manual status: {counts}")
    if args.limb_csv:
        print(f"Limb status: {table['limb_status'].value_counts().to_dict()}")
    if (table["status"] == "pending").any():
        print("PROVISIONAL: some manual labels remain pending review")
    print(f"Saved coverage and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
