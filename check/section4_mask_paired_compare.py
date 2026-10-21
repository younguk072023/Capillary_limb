"""Paired Section 4 comparison with only the segmentation mask changed.

The two inputs must use one completed outer-test MTL run, identical predicted
U/D coordinates, the same analyzer source, and the same 144 selected images.
Manual diameters come from original-image review. Only images with an approved
manual review and Success in *both* mask arms enter paired agreement metrics.
Uncertain and Failed images remain in the coverage and per-image outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


SHAPES = ("u", "tortuous", "twist")
STATUSES = {"Success", "Uncertain", "Failed"}
COORDINATES = ("Pred_U_x", "Pred_U_y", "Pred_D_x", "Pred_D_y")
OOF_COORDINATES = ("pred_U_x", "pred_U_y", "pred_D_x", "pred_D_y")
MANUAL_LINES = (
    ("B1_x1", "B1_y1", "B1_x2", "B1_y2"),
    ("B2_x1", "B2_y1", "B2_x2", "B2_y2"),
    ("U_x", "U_y", "D_x", "D_y"),
)
COMMON_EXPORT = (
    "Filename", "category", "patient_id", "fold", "Status", "Reason", "Run_ID",
    "Algorithm_SHA256", "Predicted_Mask", "Mask_SHA256", *COORDINATES,
    "Loop_Diameter_px", "Branch1_Diameter_px", "Branch2_Diameter_px",
)
GT_EXTRA = (
    "Mask_Source", "GT_Mask", "GT_Mask_SHA256", "Predicted_Mask_SHA256",
    "GT_Original", "GT_Original_SHA256",
)
OUTPUT_NAMES = (
    "section4_mask_input_pairs.csv", "section4_mask_input_summary.csv",
    "section4_mask_input_coverage.csv", "section4_mask_input_provenance.json",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(table: pd.DataFrame, names: tuple[str, ...], source: Path) -> None:
    missing = set(names) - set(table.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def require_exact_names(table: pd.DataFrame, selected: pd.DataFrame,
                        source: Path) -> pd.DataFrame:
    require_columns(table, ("Filename",), source)
    names = table["Filename"].astype(str)
    duplicated = names.str.casefold().duplicated(False)
    if duplicated.any():
        raise ValueError(f"{source}: duplicate Filename {names[duplicated].tolist()[:8]}")
    expected = set(selected["Filename"])
    actual = set(names)
    if expected != actual:
        raise ValueError(f"{source}: selection mismatch: missing={sorted(expected - actual)[:8]}, "
                         f"extra={sorted(actual - expected)[:8]}")
    return table.set_index("Filename").loc[selected["Filename"]]


def read_selection(folder: Path, expected_count: int) -> pd.DataFrame:
    rows = []
    for shape in SHAPES:
        shape_dir = folder / shape
        if not shape_dir.is_dir():
            raise FileNotFoundError(f"Missing selected category: {shape_dir}")
        for path in sorted(shape_dir.glob("*.png")):
            patient = re.match(r"^(p\d+)_", path.name, flags=re.IGNORECASE)
            if patient is None:
                raise ValueError(f"Cannot infer patient from selected image: {path.name}")
            rows.append({"Filename": path.name, "category": shape,
                         "patient_id": patient.group(1).lower(),
                         "selected_image": str(path.resolve())})
    selected = pd.DataFrame(rows)
    if selected.empty or (expected_count and len(selected) != expected_count):
        raise ValueError(f"Expected {expected_count} selected PNGs; found {len(selected)}")
    duplicates = selected["Filename"].str.casefold().duplicated(False)
    if duplicates.any():
        raise ValueError(f"Selected duplicate filenames: "
                         f"{selected.loc[duplicates, 'Filename'].tolist()[:8]}")
    return selected.sort_values(["category", "Filename"]).reset_index(drop=True)


def _positive_line(row: pd.Series, columns: tuple[str, str, str, str]) -> float:
    values = [float(row[column]) for column in columns]
    if not all(np.isfinite(values)):
        raise ValueError(f"Non-finite manual line for {row.name}: {columns}")
    width = math.dist(values[:2], values[2:])
    if width <= 0:
        raise ValueError(f"Zero-length manual line for {row.name}: {columns}")
    return width


def read_manual(path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    columns = ("Filename", "status", "patient_id", "shape", "manual_loop_px",
               "manual_thin_px", "manual_wide_px",
               *(column for line in MANUAL_LINES for column in line))
    require_columns(table, columns, path)
    table = require_exact_names(table, selected, path)
    table["status"] = table["status"].fillna("pending").str.lower().str.strip()
    if not set(table["status"]).issubset({"approved", "pending", "unmeasurable"}):
        raise ValueError(f"{path}: unexpected manual status")
    expected = selected.set_index("Filename")
    for name, row in table.iterrows():
        if str(row["patient_id"]).lower() != expected.at[name, "patient_id"]:
            raise ValueError(f"Manual patient mismatch: {name}")
        if str(row["shape"]).lower() != expected.at[name, "category"]:
            raise ValueError(f"Manual category mismatch: {name}")
        if row["status"] != "approved":
            continue
        first, second, loop = (_positive_line(row, line) for line in MANUAL_LINES)
        for column, value in (("manual_loop_px", loop),
                              ("manual_thin_px", min(first, second)),
                              ("manual_wide_px", max(first, second))):
            recorded = float(row[column])
            if not math.isfinite(recorded) or abs(recorded - value) > 0.02:
                raise ValueError(f"Stale manual {column}: {name}")
    return table


def read_oof(path: Path, selected: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    manifest = json.loads((path / "experiment_manifest.json").read_text(encoding="utf-8-sig"))
    run_id = str(manifest.get("run_id", ""))
    if not run_id:
        raise ValueError(f"MTL run has no run_id: {path}")
    chosen = set(selected["Filename"])
    selected_patient = selected.set_index("Filename")["patient_id"].to_dict()
    rows = []
    for fold_dir in sorted(path.glob("fold_*")):
        if not fold_dir.is_dir() or not re.fullmatch(r"fold_\d+", fold_dir.name):
            continue
        fold = int(fold_dir.name.split("_")[-1])
        split_path = fold_dir / "patient_split.csv"
        coords_path = fold_dir / "predicted_coordinates.csv"
        split = pd.read_csv(split_path)
        coords = pd.read_csv(coords_path)
        require_columns(split, ("filename", "patient_id", "split"), split_path)
        require_columns(coords, ("filename", "fold", *OOF_COORDINATES,
                                 "pred_length_px"), coords_path)
        if split["filename"].duplicated().any() or coords["filename"].duplicated().any():
            raise ValueError(f"Duplicate OOF filenames in {fold_dir}")
        test = split.loc[split["split"] == "outer_test"]
        if set(test["patient_id"]) & set(split.loc[split["split"] != "outer_test", "patient_id"]):
            raise ValueError(f"Patient split leakage in {fold_dir}")
        joined = test.loc[test["filename"].isin(chosen)].merge(
            coords, on="filename", how="left", validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"Missing selected OOF coordinates in {fold_dir}")
        for _, row in joined.iterrows():
            name = str(row["filename"])
            if str(row["patient_id"]).lower() != selected_patient[name] or int(row["fold"]) != fold:
                raise ValueError(f"OOF patient/fold mismatch: {name}")
            mask = fold_dir / "predicted_masks" / f"{Path(name).stem}.png"
            if not mask.is_file():
                raise FileNotFoundError(mask)
            uv = [float(row[column]) for column in OOF_COORDINATES]
            if not all(np.isfinite(uv)) or math.dist(uv[:2], uv[2:]) <= 0:
                raise ValueError(f"Invalid OOF U/D: {name}")
            if abs(float(row["pred_length_px"]) - math.dist(uv[:2], uv[2:])) > 0.02:
                raise ValueError(f"OOF loop/U-D mismatch: {name}")
            rows.append({"Filename": name, "fold": fold,
                         "oof_mask": str(mask.resolve()), "oof_mask_hash": file_hash(mask),
                         **dict(zip(COORDINATES, uv)),
                         "Loop_Diameter_px": math.dist(uv[:2], uv[2:])})
    table = pd.DataFrame(rows)
    return run_id, require_exact_names(table, selected, path)


def read_gt_manifest(path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(table, ("Category", "Selected_Image", "GT_Original", "GT_Copy",
                            "Original_SHA256", "Copy_SHA256"), path)
    table["Filename"] = table["Selected_Image"].map(lambda value: Path(str(value)).name)
    table = require_exact_names(table, selected, path)
    reference = selected.set_index("Filename")
    for name, row in table.iterrows():
        if str(row["Category"]).lower() != reference.at[name, "category"]:
            raise ValueError(f"GT category mismatch: {name}")
        if Path(row["Selected_Image"]).resolve() != Path(reference.at[name, "selected_image"]):
            raise ValueError(f"GT selection path mismatch: {name}")
        copied = Path(row["GT_Copy"])
        if not copied.is_file() or file_hash(copied).lower() != str(row["Copy_SHA256"]).lower():
            raise ValueError(f"GT copied mask hash mismatch: {name}")
        if str(row["Original_SHA256"]).lower() != str(row["Copy_SHA256"]).lower():
            raise ValueError(f"GT copy differs from recorded source: {name}")
    return table


def _same_float(left: object, right: object, tolerance: float = 1e-5) -> bool:
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance


def read_export(path: Path, selected: pd.DataFrame, oof: pd.DataFrame,
                run_id: str, gt_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(table, COMMON_EXPORT + (GT_EXTRA if gt_manifest is not None else ()), path)
    table = require_exact_names(table, selected, path)
    if set(table["Status"]) - STATUSES:
        raise ValueError(f"{path}: unknown algorithm statuses")
    if table["Algorithm_SHA256"].isna().any() or table["Algorithm_SHA256"].nunique() != 1:
        raise ValueError(f"{path}: algorithm hash is missing or inconsistent")
    selected_index = selected.set_index("Filename")
    for name, row in table.iterrows():
        if str(row["Run_ID"]) != run_id or int(row["fold"]) != int(oof.at[name, "fold"]):
            raise ValueError(f"Run/fold mismatch: {name}")
        if str(row["category"]).lower() != selected_index.at[name, "category"] or \
                str(row["patient_id"]).lower() != selected_index.at[name, "patient_id"]:
            raise ValueError(f"Export category/patient mismatch: {name}")
        if Path(row["Predicted_Mask"]).resolve() != Path(oof.at[name, "oof_mask"]):
            raise ValueError(f"Predicted-mask provenance mismatch: {name}")
        expected_oof_hash = oof.at[name, "oof_mask_hash"]
        if gt_manifest is None:
            if str(row["Mask_SHA256"]).lower() != expected_oof_hash:
                raise ValueError(f"Predicted mask hash mismatch: {name}")
        else:
            manifest = gt_manifest.loc[name]
            if str(row["Mask_Source"]) != "GT_Copy":
                raise ValueError(f"GT mask source mismatch: {name}")
            if Path(row["GT_Mask"]).resolve() != Path(manifest["GT_Copy"]).resolve() or \
                    Path(row["GT_Original"]).resolve() != Path(manifest["GT_Original"]).resolve():
                raise ValueError(f"GT mask path mismatch: {name}")
            gt_hash = str(manifest["Copy_SHA256"]).lower()
            if str(row["Mask_SHA256"]).lower() != gt_hash or \
                    str(row["GT_Mask_SHA256"]).lower() != gt_hash or \
                    str(row["GT_Original_SHA256"]).lower() != str(manifest["Original_SHA256"]).lower() or \
                    str(row["Predicted_Mask_SHA256"]).lower() != expected_oof_hash:
                raise ValueError(f"GT or OOF mask hash mismatch: {name}")
        for column in (*COORDINATES, "Loop_Diameter_px"):
            if not _same_float(row[column], oof.at[name, column]):
                raise ValueError(f"Predicted U/D changed in {path}: {name}, {column}")
        if row["Status"] == "Success":
            widths = [float(row[f"Branch{number}_Diameter_px"]) for number in (1, 2)]
            if not all(math.isfinite(value) and value > 0 for value in widths):
                raise ValueError(f"Success without two valid widths: {name}")
        elif not str(row["Reason"]).strip() or pd.isna(row["Reason"]):
            raise ValueError(f"{row['Status']} without reason: {name}")
    return table


def _reported_widths(row: pd.Series) -> tuple[float, float]:
    widths = pd.to_numeric(pd.Series([row["Branch1_Diameter_px"],
                                      row["Branch2_Diameter_px"]]), errors="coerce")
    if len(widths) != 2 or not np.isfinite(widths).all() or (widths <= 0).any():
        return math.nan, math.nan
    return float(widths.min()), float(widths.max())


def make_pairs(selected: pd.DataFrame, manual: pd.DataFrame,
               predicted: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in selected.itertuples(index=False):
        name = item.Filename
        m, p, g = manual.loc[name], predicted.loc[name], gt.loc[name]
        pthin, pwide = _reported_widths(p)
        gthin, gwide = _reported_widths(g)
        common = m["status"] == "approved" and p["Status"] == g["Status"] == "Success"
        row = {"Filename": name, "category": item.category, "patient_id": item.patient_id,
               "fold": int(p["fold"]), "manual_status": m["status"],
               "pred_status": p["Status"], "gt_status": g["Status"],
               "pred_reason": p["Reason"] if pd.notna(p["Reason"]) else "",
               "gt_reason": g["Reason"] if pd.notna(g["Reason"]) else "",
               "common_success": common,
               "manual_loop_px": m["manual_loop_px"],
               "manual_thin_px": m["manual_thin_px"],
               "manual_wide_px": m["manual_wide_px"],
               "pred_loop_px": float(p["Loop_Diameter_px"]),
               "gt_loop_px": float(g["Loop_Diameter_px"]),
               "pred_reported_thin_px": pthin, "pred_reported_wide_px": pwide,
               "gt_reported_thin_px": gthin, "gt_reported_wide_px": gwide,
               "pred_mask": p["Predicted_Mask"], "gt_mask": g["GT_Mask"]}
        for target, mp, pp, gp in (("thin", "manual_thin_px", pthin, gthin),
                                   ("wide", "manual_wide_px", pwide, gwide)):
            row[f"pred_{target}_px"] = pp if common else math.nan
            row[f"gt_{target}_px"] = gp if common else math.nan
            row[f"pred_abs_error_{target}_px"] = abs(pp - float(m[mp])) if common else math.nan
            row[f"gt_abs_error_{target}_px"] = abs(gp - float(m[mp])) if common else math.nan
            row[f"gt_error_reduction_{target}_px"] = (
                row[f"pred_abs_error_{target}_px"] - row[f"gt_abs_error_{target}_px"])
            row[f"pred_minus_gt_{target}_px"] = pp - gp if common else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def icc_pair(manual: np.ndarray, automated: np.ndarray) -> tuple[float, float]:
    """Two-way single-measure absolute agreement ICC(A,1) and ICC(C,1)."""
    x = np.column_stack((manual, automated)).astype(float)
    n, k = x.shape
    if n < 3:
        return math.nan, math.nan
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = x - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator_c = ms_rows + (k - 1) * ms_error
    denominator_a = denominator_c + k * (ms_cols - ms_error) / n
    agreement = (ms_rows - ms_error) / denominator_a if denominator_a else math.nan
    consistency = (ms_rows - ms_error) / denominator_c if denominator_c else math.nan
    return float(agreement), float(consistency)


def arm_metrics(manual: np.ndarray, automated: np.ndarray) -> dict[str, float]:
    if len(manual) < 3:
        return {}
    difference = automated - manual
    agreement, consistency = icc_pair(manual, automated)
    sd = float(np.std(difference, ddof=1))
    return {"pearson_r": float(np.corrcoef(manual, automated)[0, 1])
            if np.ptp(manual) and np.ptp(automated) else math.nan,
            "icc_absolute_A1": agreement,
            "icc_consistency_C1": consistency,
            "bias_px": float(np.mean(difference)),
            "loa_lower_px": float(np.mean(difference) - 1.96 * sd),
            "loa_upper_px": float(np.mean(difference) + 1.96 * sd),
            "mae_px": float(np.mean(np.abs(difference)))}


def _paired_statistics(manual: np.ndarray, pred: np.ndarray,
                       gt: np.ndarray) -> dict[str, float]:
    improvement = np.abs(pred - manual) - np.abs(gt - manual)
    direct = pred - gt
    result = {"mae_improvement_gt_px": float(np.mean(improvement)),
              "median_abs_error_improvement_gt_px": float(np.median(improvement)),
              "pred_minus_gt_mean_px": float(np.mean(direct)),
              "pred_minus_gt_abs_mean_px": float(np.mean(np.abs(direct)))}
    for prefix, automated in (("pred", pred), ("gt", gt)):
        result.update({f"{prefix}_{key}": value
                       for key, value in arm_metrics(manual, automated).items()})
    return result


def _cluster_intervals(part: pd.DataFrame, manual: np.ndarray, pred: np.ndarray,
                       gt: np.ndarray, repetitions: int) -> dict[str, float]:
    if repetitions <= 0 or part["patient_id"].nunique() < 3:
        return {}
    patients = part["patient_id"].to_numpy()
    unique = np.unique(patients)
    positions = {patient: np.flatnonzero(patients == patient) for patient in unique}
    rng = np.random.default_rng(20260928)
    distributions: dict[str, list[float]] = {}
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[patient] for patient in sampled])
        if len(indices) < 3:
            continue
        for key, value in _paired_statistics(manual[indices], pred[indices], gt[indices]).items():
            if np.isfinite(value):
                distributions.setdefault(key, []).append(value)
    result = {}
    for key, values in distributions.items():
        if len(values) >= max(50, repetitions // 2):
            result[f"{key}_ci95_low"] = float(np.percentile(values, 2.5))
            result[f"{key}_ci95_high"] = float(np.percentile(values, 97.5))
    return result


def make_summary(pairs: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    for shape in ("all", *SHAPES):
        group = pairs if shape == "all" else pairs.loc[pairs["category"] == shape]
        common = group.loc[group["common_success"]]
        for target in ("thin", "wide"):
            result = {"target": target, "shape": shape,
                      "selected_n": len(group),
                      "manual_approved_n": int(group["manual_status"].eq("approved").sum()),
                      "pred_success_n": int(group["pred_status"].eq("Success").sum()),
                      "gt_success_n": int(group["gt_status"].eq("Success").sum()),
                      "common_success_n": len(common),
                      "patient_n": common["patient_id"].nunique()}
            if len(common) >= 3:
                manual = common[f"manual_{target}_px"].to_numpy(dtype=float)
                pred = common[f"pred_{target}_px"].to_numpy(dtype=float)
                gt = common[f"gt_{target}_px"].to_numpy(dtype=float)
                result.update(_paired_statistics(manual, pred, gt))
                result.update(_cluster_intervals(common, manual, pred, gt, repetitions))
            rows.append(result)
    return pd.DataFrame(rows)


def make_coverage(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shape in ("all", *SHAPES):
        group = pairs if shape == "all" else pairs.loc[pairs["category"] == shape]
        for (pred_status, gt_status), part in group.groupby(
                ["pred_status", "gt_status"], sort=True):
            rows.append({"shape": shape, "selected_n": len(group),
                         "pred_status": pred_status, "gt_status": gt_status,
                         "count": len(part), "patient_n": part["patient_id"].nunique()})
    return pd.DataFrame(rows)


def compare(selection_dir: Path, manual_csv: Path, pred_csv: Path,
            gt_csv: Path, gt_manifest_csv: Path, oof_root: Path,
            expected_count: int = 144, bootstrap_reps: int = 1000
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    selected = read_selection(selection_dir, expected_count)
    manual = read_manual(manual_csv, selected)
    run_id, oof = read_oof(oof_root, selected)
    gt_manifest = read_gt_manifest(gt_manifest_csv, selected)
    predicted = read_export(pred_csv, selected, oof, run_id)
    gt = read_export(gt_csv, selected, oof, run_id, gt_manifest)
    if predicted["Algorithm_SHA256"].iat[0] != gt["Algorithm_SHA256"].iat[0]:
        raise ValueError("Predicted and GT exports used different algorithm source hashes")
    for column in (*COORDINATES, "Loop_Diameter_px"):
        if not np.allclose(predicted[column].astype(float), gt[column].astype(float),
                           atol=1e-5, rtol=0):
            raise ValueError(f"Predicted U/D differs between mask arms: {column}")
    pairs = make_pairs(selected, manual, predicted, gt)
    summary = make_summary(pairs, bootstrap_reps)
    coverage = make_coverage(pairs)
    provenance = {"run_id": run_id,
                  "algorithm_sha256": str(predicted["Algorithm_SHA256"].iat[0]),
                  "selected_n": len(selected),
                  "patient_n": selected["patient_id"].nunique(),
                  "common_success_n": int(pairs["common_success"].sum()),
                  "bootstrap_reps": bootstrap_reps,
                  "selection_dir": str(selection_dir.resolve()),
                  "manual_csv": str(manual_csv.resolve()),
                  "manual_csv_sha256": file_hash(manual_csv),
                  "pred_csv": str(pred_csv.resolve()),
                  "pred_csv_sha256": file_hash(pred_csv),
                  "gt_csv": str(gt_csv.resolve()),
                  "gt_csv_sha256": file_hash(gt_csv),
                  "gt_manifest_csv": str(gt_manifest_csv.resolve()),
                  "gt_manifest_sha256": file_hash(gt_manifest_csv),
                  "oof_root": str(oof_root.resolve())}
    return pairs, summary, coverage, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--manual-csv", required=True, type=Path)
    parser.add_argument("--pred-csv", required=True, type=Path)
    parser.add_argument("--gt-csv", required=True, type=Path)
    parser.add_argument("--gt-manifest", required=True, type=Path)
    parser.add_argument("--oof-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=144)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_count < 0 or args.bootstrap_reps < 0:
        parser.error("--expected-count and --bootstrap-reps must be nonnegative")
    output = args.output_dir.resolve()
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not args.overwrite:
        parser.error(f"Output exists; use --overwrite: {existing}")
    pairs, summary, coverage, provenance = compare(
        args.selection_dir, args.manual_csv, args.pred_csv, args.gt_csv,
        args.gt_manifest, args.oof_root, args.expected_count, args.bootstrap_reps)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / OUTPUT_NAMES[0], index=False, encoding="utf-8-sig")
    summary.to_csv(output / OUTPUT_NAMES[1], index=False, encoding="utf-8-sig")
    coverage.to_csv(output / OUTPUT_NAMES[2], index=False, encoding="utf-8-sig")
    (output / OUTPUT_NAMES[3]).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Compared {len(pairs)} selected images across {provenance['patient_n']} patients; "
          f"{provenance['common_success_n']} approved/common-Success pairs")
    print(f"Saved paired comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
