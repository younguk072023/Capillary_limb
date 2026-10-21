r"""Compare manual diameters with selected five-fold U-Net MTL predictions.

This final evaluator accepts only DINOv3 runs. The selected masks in
dinov3_segmentation/{U,tortuous,crossing} must match the
corresponding outer-test fold PNGs byte for byte. Branch diameters are measured
afresh from those masks with the current Capillary_limb analyzer. Loop diameter
comes from the same fold's predicted U/D coordinates, not from a mask.
Title-free scatter and Bland-Altman figures are saved for each site in µm
(default: 0.55 µm/px), as PNG and TIFF, with a combined overview PNG.
Original pixel CSV columns are retained alongside calibrated *_um columns.

PowerShell:
    & C:\Users\USER\.conda\envs\younguk\python.exe -B C:\Users\USER\Desktop\Capillary_limb\limb\compare_predicted_masks.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

DESKTOP = Path.home() / "Desktop"
LIMB_DIR = DESKTOP / "Capillary_limb" / "limb"
COMMON_DIR = Path(__file__).resolve().parent
if not (COMMON_DIR / "compare_measurements.py").is_file():
    COMMON_DIR = LIMB_DIR
sys.path.insert(0, str(COMMON_DIR))
import compare_measurements as common  # noqa: E402

np = common.np
pd = common.pd
oof = common.oof
validation = common.validation

DEFAULT_MASKS = DESKTOP / "dinov3_segmentation"
SHAPE_FOLDERS = {"u": "U", "tortuous": "tortuous", "twist": "crossing"}
MEASURES = (
    ("loop", "Loop diameter", "manual_loop_px", "auto_loop_px", "predicted_ud"),
    ("thin_branch", "Thin branch (estimated arterial side)",
     "manual_thin_px", "auto_thin_px", "predicted_mask"),
    ("wide_branch", "Wide branch (estimated venous side)",
     "manual_wide_px", "auto_wide_px", "predicted_mask"),
)
MANIFEST_FIELDS = ("Filename", "shape", "fold", "Selected_Mask",
                   "Selected_Mask_SHA256", "OOF_Mask", "OOF_Mask_SHA256",
                   "Root_Mask", "Root_Mask_SHA256")


def next_output(requested: Path | None, encoder_name: str, selected_n: int) -> Path:
    if requested is not None:
        output = requested.resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        return output
    default_output = (DESKTOP / "Multi-Task-Learning" / "outputs" /
                      f"predicted_mask_measurements_{encoder_name}_{selected_n}_{datetime.now():%Y%m%d}")
    output = default_output.resolve()
    counter = 2
    while output.exists():
        output = default_output.with_name(
            f"{default_output.name}_run{counter}").resolve()
        counter += 1
    return output


def verify_selected_masks(mask_dir: Path, prepared: list[oof.PreparedImage]
                          ) -> tuple[list[oof.PreparedImage], list[dict[str, object]]]:
    expected = {folder: set() for folder in SHAPE_FOLDERS.values()}
    for item in prepared:
        expected[SHAPE_FOLDERS[item.selected.category.lower()]].add(
            f"{Path(item.selected.filename).stem}.png")
    for folder, names in expected.items():
        directory = mask_dir / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing prediction folder: {directory}")
        actual = {path.name for path in directory.iterdir() if path.is_file()}
        if actual != names:
            raise ValueError(f"Prediction inventory mismatch in {directory}: "
                             f"missing={sorted(names - actual)[:8]}, "
                             f"extra={sorted(actual - names)[:8]}; "
                             f"expected={len(names)}")

    selected_prepared = []
    manifest = []
    for item in prepared:
        name = f"{Path(item.selected.filename).stem}.png"
        folder = SHAPE_FOLDERS[item.selected.category.lower()]
        selected_mask = (mask_dir / folder / name).resolve()
        root_mask = (mask_dir / name).resolve()
        if not root_mask.is_file():
            raise FileNotFoundError(f"Missing root prediction mask: {root_mask}")
        source_hash = oof.file_sha256(item.mask_path)
        selected_hash = oof.file_sha256(selected_mask)
        root_hash = oof.file_sha256(root_mask)
        if source_hash != selected_hash or selected_hash != root_hash:
            raise ValueError(f"Selected/root/outer-test prediction differs: {name}")
        size = oof.png_size(selected_mask)
        if size != (item.mask_width, item.mask_height):
            raise ValueError(f"Prediction frame differs from U/D coordinates: {name}")
        selected_prepared.append(replace(item, mask_path=selected_mask))
        manifest.append({
            "Filename": item.selected.filename,
            "shape": folder,
            "fold": item.fold,
            "Selected_Mask": str(selected_mask),
            "Selected_Mask_SHA256": selected_hash,
            "OOF_Mask": str(item.mask_path.resolve()),
            "OOF_Mask_SHA256": source_hash,
            "Root_Mask": str(root_mask),
            "Root_Mask_SHA256": root_hash,
        })
    if len(selected_prepared) != len(prepared) or len(manifest) != len(prepared):
        raise ValueError("Selected prediction masks do not match the cohort")
    return selected_prepared, manifest


def build_comparison(selected: pd.DataFrame, manual: pd.DataFrame,
                     measurements: list[dict[str, object]],
                     manifest: list[dict[str, object]]) -> pd.DataFrame:
    hashes = {str(row["Filename"]): str(row["Selected_Mask_SHA256"])
              for row in manifest}
    rows = []
    for record in measurements:
        name = str(record["Filename"])
        if name not in hashes or record["Mask_SHA256"] != hashes[name]:
            raise ValueError(f"Exported mask hash differs from verified input: {name}")
        b1 = pd.to_numeric(record["Branch1_Diameter_px"], errors="coerce")
        b2 = pd.to_numeric(record["Branch2_Diameter_px"], errors="coerce")
        measured = bool(np.isfinite(b1) and np.isfinite(b2) and b1 > 0 and b2 > 0)
        loop = pd.to_numeric(record["Loop_Diameter_px"], errors="coerce")
        rows.append({
            "Filename": name, "fold": int(record["fold"]),
            "run_id": record["Run_ID"],
            "predicted_mask": record["Predicted_Mask"],
            "predicted_mask_sha256": record["Mask_SHA256"],
            "algorithm_sha256": record["Algorithm_SHA256"],
            "pred_status": record["Status"],
            "pred_reason": record["Reason"],
            "auto_loop_px": loop,
            "pred_branch1_px": b1, "pred_branch2_px": b2,
            "auto_thin_px": min(b1, b2) if measured else np.nan,
            "auto_wide_px": max(b1, b2) if measured else np.nan,
            "pred_measured_pair": measured,
        })
    measured_table = pd.DataFrame(rows)
    table = selected[["Filename", "shape", "patient_id"]].merge(
        manual, on="Filename", how="left", validate="one_to_one")
    table = table.merge(measured_table, on="Filename", how="left",
                        validate="one_to_one", indicator=True)
    if (len(table) != len(selected) or not table["_merge"].eq("both").all() or
            not table["Filename"].is_unique):
        raise ValueError("Predicted mask measurements do not match the selected cohort")
    table = table.drop(columns="_merge")
    if not table["pred_status"].isin(["Success", "Uncertain", "Failed"]).all():
        raise ValueError("Unexpected analyzer QC status")
    for _, _, manual_col, auto_col, _ in MEASURES:
        table[auto_col.replace("auto_", "error_")] = table[auto_col] - table[manual_col]
    table["shape_report"] = table["shape"].replace({"twist": "crossing"})
    return table


def summarize(table: pd.DataFrame, shape: str = "all") -> pd.DataFrame:
    subset = table if shape == "all" else table.loc[table["shape_report"] == shape]
    results = []
    qc_counts = subset["pred_status"].value_counts()
    for measure, label, manual_col, auto_col, source in MEASURES:
        manual = pd.to_numeric(subset[manual_col], errors="coerce")
        auto = pd.to_numeric(subset[auto_col], errors="coerce")
        eligible = (subset["status"].eq("approved") & np.isfinite(manual) &
                    np.isfinite(auto) & (manual > 0) & (auto > 0))
        checked = subset.loc[eligible]
        if len(checked) < 3:
            raise ValueError(f"Too few valid {shape}/{measure} measurements")
        metrics = validation._point_metrics(
            manual.loc[eligible].to_numpy(dtype=float),
            auto.loc[eligible].to_numpy(dtype=float))
        results.append({
            "shape": shape, "measure": measure, "label": label,
            "source": source, "n": len(checked), "selected_n": len(subset),
            "patient_n": int(checked["patient_id"].nunique()),
            "success_n": int(qc_counts.get("Success", 0)),
            "uncertain_n": int(qc_counts.get("Uncertain", 0)),
            "failed_n": int(qc_counts.get("Failed", 0)),
            "inclusion_rule": "approved + finite positive measured values, any QC status",
            **metrics,
        })
    return pd.DataFrame(results)


def execute(args: argparse.Namespace) -> Path:
    print(f"Evaluation script: {Path(__file__).resolve()}", flush=True)
    run_dir = args.run_dir.resolve()
    expected_run_id, encoder_name = common.run_model_info(run_dir)
    selection_dir = args.selection_dir.resolve()
    selection_csv = args.selection_csv.resolve()
    mask_dir = args.mask_dir.resolve()
    analyzer_dir = args.analyzer_dir.resolve()
    manual_csv = args.manual_csv.resolve()
    selection_hash = oof.file_sha256(selection_csv)
    selected = oof.select_images(None, selection_csv, args.expected_count)
    selected_n = len(selected)
    output = next_output(args.output_dir, encoder_name, selected_n)
    print(f"Planned output directory: {output}", flush=True)
    prepared, run_id = oof.verify_run(run_dir, selected)
    if run_id != expected_run_id or len({item.fold for item in prepared}) != 5:
        raise ValueError("Run ID or selected fold coverage mismatch")
    master_path = common.verify_master(run_dir, prepared)
    manual_hash = oof.file_sha256(manual_csv)
    selection_table, manual = common.validate_selection(
        selection_dir, manual_csv, selected)
    original_selection = common.read_original_selection(selection_dir)
    excluded_names = sorted(set(original_selection["Filename"]) -
                            set(selection_table["Filename"]))
    predicted, manifest = verify_selected_masks(mask_dir, prepared)
    common.print_input_summary(run_dir, encoder_name, selected_n, selection_csv,
                               manual_csv, mask_dir, "Predicted")
    print(f"Verified {len(predicted)} selected masks against five OOF folds; "
          f"run={run_id}", flush=True)
    if getattr(args, "dry_run", False):
        print("Input check passed; no measurements or output files were written.", flush=True)
        return output
    measurements = oof.export_rows(predicted, run_id, analyzer_dir)
    if len(measurements) != selected_n:
        raise ValueError(f"Analyzer did not return {selected_n} rows")
    comparison = build_comparison(selection_table, manual, measurements, manifest)
    summary = summarize(comparison)
    by_shape = pd.concat([summarize(comparison, name)
                          for name in ("u", "tortuous", "crossing")],
                         ignore_index=True)
    table_12 = common.make_agreement_table_12(summary, by_shape)
    comparison = common.add_micrometer_columns(comparison, args.um_per_pixel)
    summary = common.add_micrometer_columns(summary, args.um_per_pixel)
    by_shape = common.add_micrometer_columns(by_shape, args.um_per_pixel)
    stat_columns = ("pearson_r", "icc_absolute_A1", "icc_consistency_C1",
                    "bias_px", "loa_lower_px", "loa_upper_px", "mae_px")
    if not np.isfinite(pd.concat([summary, by_shape])[list(stat_columns)]
                       .to_numpy(dtype=float)).all():
        raise ValueError("Non-finite summary statistic")
    if oof.file_sha256(manual_csv) != manual_hash:
        raise RuntimeError("Manual CSV changed during analysis; rerun")
    if oof.file_sha256(selection_csv) != selection_hash:
        raise RuntimeError("Cohort CSV changed during analysis; rerun")
    for record in manifest:
        path = Path(str(record["Selected_Mask"]))
        if oof.file_sha256(path) != record["Selected_Mask_SHA256"]:
            raise RuntimeError(f"Predicted mask changed during analysis: {path}")
    algorithm_hash = measurements[0]["Algorithm_SHA256"]
    if (any(row["Algorithm_SHA256"] != algorithm_hash for row in measurements) or
            oof.analyzer_sha256(analyzer_dir) != algorithm_hash):
        raise RuntimeError("Analyzer source changed during analysis; rerun")

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
        shutil.copyfile(selection_csv, stage / f"diameter_selection_{selected_n}.csv")
        common.write_manual_subset_exports(stage, manual_csv, selected)
        with (stage / f"predicted_measurements_{selected_n}.csv").open(
                "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=oof.FIELDNAMES)
            writer.writeheader()
            writer.writerows(measurements)
        with (stage / f"predicted_mask_manifest_{selected_n}.csv").open(
                "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest)
        figure_files = common.draw_figures(comparison, summary, stage)
        metadata = {
            **common.figure_provenance(args.um_per_pixel, figure_files),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "design": f"Patient-held-out 5-fold U-Net MTL with {common.ENCODER_LABELS[encoder_name]} encoder",
            "encoder_name": encoder_name,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "master_predicted_coordinates": str(master_path),
            "master_predicted_coordinates_sha256": oof.file_sha256(master_path),
            "selection_dir": str(selection_dir),
            "selection_csv": str(selection_csv),
            "selection_csv_sha256": selection_hash,
            "original_selected_n": len(original_selection),
            "excluded_n": len(excluded_names),
            "excluded_filenames": excluded_names,
            "exclusion_source": "Explicit user-selected cohort CSV; original image and manual archives retained",
            "manual_csv": str(manual_csv),
            "manual_csv_sha256": manual_hash,
            "selected_predicted_mask_dir": str(mask_dir),
            "mask_matching": f"{selected_n} selected PNGs byte-identical to root PNGs and matching outer-test fold PNGs",
            "analyzer_dir": str(analyzer_dir),
            "algorithm_sha256": algorithm_hash,
            "selected_n": len(comparison),
            "pred_qc_counts": {str(k): int(v) for k, v in
                               comparison["pred_status"].value_counts().items()},
            "measured_pair_n": int(comparison["pred_measured_pair"].sum()),
            "branch_identity": "Thin/estimated arterial and wide/estimated venous are width-ranked proxies, not independently confirmed anatomical identities.",
            "inclusion_rule": "All approved images with finite positive measurements, including analyzer QC Uncertain",
            "loop_source": "Predicted U/D distance from master coordinates, verified against the corresponding outer-test fold",
            "branch_source": "Current limb analyzer run on selected prediction PNGs using the same predicted U/D coordinates",
            "agreement_table_12_rows": len(table_12),
        }
        (stage / "provenance.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Output appeared during analysis: {output}")
        if oof.file_sha256(manual_csv) != manual_hash:
            raise RuntimeError("Manual CSV changed while writing results; rerun")
        if oof.file_sha256(selection_csv) != selection_hash:
            raise RuntimeError("Cohort CSV changed while writing results; rerun")
        shutil.copytree(stage, output)
    print(f"Saved predicted-mask comparison to {output}", flush=True)
    print(f"Saved {len(figure_files)} figures in µm (1 px = {args.um_per_pixel:g} µm); difference = automatic - manual.", flush=True)
    console = table_12.rename(columns={"형태": "Shape", "측정 부위": "Measure"})
    console["Measure"] = console["Measure"].replace({
        "루프": "Loop", "세동맥측": "Thin branch", "세정맥측": "Wide branch",
    })
    counts = pd.concat([by_shape, summary], ignore_index=True)
    count_map = {(row["shape"], row["measure"]): int(row["n"])
                 for _, row in counts.iterrows()}
    console.insert(2, "N", [count_map[(shape, measure)]
                            for shape, _ in common.TABLE_SHAPES
                            for measure, _ in common.TABLE_MEASURES])
    print(console.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
          flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=common.DEFAULT_RUN)
    parser.add_argument("--selection-dir", type=Path,
                        default=common.DEFAULT_SELECTION)
    parser.add_argument("--selection-csv", type=Path,
                        default=common.DEFAULT_SELECTION_CSV,
                        help="Explicit cohort shared by GT and prediction evaluation")
    parser.add_argument("--expected-count", type=int, default=141)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--manual-csv", type=Path,
                        default=common.DEFAULT_MANUAL)
    # parser.add_argument(
    # "--manual-csv",
    # type=Path,
    # default=(
    #     DESKTOP / "Capillary_limb" / "label" / "outputs"
    #     / "section4_review_pa_141_rater2"
    #     / "manual_review_141_rater2.csv"
    # ),
#)

    ########## 위코드 측정자 2
    parser.add_argument("--analyzer-dir", type=Path,
                        default=common.DEFAULT_ANALYZER)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate model, paths, cohort, manual values and masks without writing results")
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh output directory (default: next unused numbered run)")
    parser.add_argument("--um-per-pixel", type=float, default=common.DEFAULT_UM_PER_PIXEL,
                        help="Micrometers per pixel for figures and *_um CSV columns (default: 0.55)")
    args = parser.parse_args()
    if not np.isfinite(args.um_per_pixel) or args.um_per_pixel <= 0:
        parser.error("--um-per-pixel must be a finite positive number")
    execute(args)


if __name__ == "__main__":
    main()
