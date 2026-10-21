r"""Measure the same 144 ROIs on ResNet50 OOF and GT segmentation masks.

Predicted U/D points come from the completed patient-held-out five-fold MTL
run in both arms. The predicted masks are read from the category folders in
Resnet50_segmentation; GT TIFFs are read from GT_masks_144. The GT manifest's
old GT_Copy paths are rebased into a new copy in the result directory.
Primary branch agreement includes every approved image with two finite,
positive measured branch widths, regardless of the analyzer's QC status.
Success-only branch agreement is saved separately as a sensitivity analysis.

Usage (PowerShell):
    & C:\Users\USER\.conda\envs\younguk\python.exe -B C:\Users\USER\Desktop\Capillary_limb\check\run_section4_resnet50_144.py --dry-run
    & C:\Users\USER\.conda\envs\younguk\python.exe -B C:\Users\USER\Desktop\Capillary_limb\check\run_section4_resnet50_144.py --output-dir C:\Users\USER\Desktop\Multi-Task-Learning\outputs\section4_resnet50_144_new_run
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile


DESKTOP = Path.home() / "Desktop"
MTL_ROOT = DESKTOP / "Multi-Task-Learning"
LIMB_ROOT = DESKTOP / "Capillary_limb"
CHECK_DIR = LIMB_ROOT / "check"
RUN_DIR = DESKTOP / "논문 결과" / "results_unet_mtl_resnet50_heatmap_bce_dice_fe62dc8de69f"
SELECTION_DIR = DESKTOP / "Pa_capillary"
PRED_DIR = DESKTOP / "Resnet50_segmentation"
GT_DIR = DESKTOP / "GT_masks_144"
MANUAL_CSV = LIMB_ROOT / "label" / "outputs" / "section4_review_pa_144" / "manual_review_144.csv"
OUTPUT_DIR = MTL_ROOT / "outputs" / "section4_resnet50_144_all_measured"
CATEGORY_TO_PRED = {"U": "U", "tortuous": "tortuous", "twist": "crossing"}
GT_MANIFEST_COLUMNS = (
    "Category", "Selected_Image", "GT_Original", "GT_Copy",
    "Original_SHA256", "Copy_SHA256", "Bytes",
)

sys.dont_write_bytecode = True
sys.path.insert(0, str(CHECK_DIR))
import section4_oof_limb_export as base  # noqa: E402
import section4_gt_limb_export as gt_export  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_inputs(args: argparse.Namespace):
    manifest = json.loads((args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8-sig"))
    spec = manifest["run_spec"]
    experiment = spec["experiment"]
    if (spec["n_splits"] != 5 or experiment["model_name"] != "unet" or
            experiment["encoder_name"] != "resnet50" or
            experiment["task_mode"] != "mtl"):
        base.fail("The completed run must be five-fold U-Net MTL with a ResNet50 encoder")
    selected = base.select_images(args.selection_dir, None, 144)
    prepared, run_id = base.verify_run(args.run_dir, selected)
    if not run_id:
        base.fail("Completed run has no run_id")
    grouped = []
    for item in prepared:
        category = item.selected.category
        if category not in CATEGORY_TO_PRED:
            base.fail(f"Unexpected selection category: {category}")
        name = Path(item.selected.filename).stem + ".png"
        flat = args.pred_dir / name
        mask = args.pred_dir / CATEGORY_TO_PRED[category] / name
        canonical_hash = base.file_sha256(item.mask_path)
        if not flat.is_file() or not mask.is_file():
            base.fail(f"Missing flat or grouped prediction: {item.selected.filename}")
        if (base.file_sha256(flat) != canonical_hash or
                base.file_sha256(mask) != canonical_hash):
            base.fail(f"Flat/grouped mask differs from its OOF fold: {item.selected.filename}")
        if base.png_size(mask) != (item.mask_width, item.mask_height):
            base.fail(f"Grouped mask dimensions changed: {mask}")
        new_item = replace(item, mask_path=mask)
        base.finite_coordinate_values(new_item)
        grouped.append(new_item)
    return selected, grouped, run_id


def rebase_gt_manifest(gt_dir: Path, output: Path) -> None:
    original = gt_dir / "gt_mask_manifest_144.csv"
    rows = base.csv_rows(original, GT_MANIFEST_COLUMNS)
    if len(rows) != 144:
        base.fail(f"Expected 144 GT manifest rows, found {len(rows)}")
    for row in rows:
        category = row["Category"]
        if category not in CATEGORY_TO_PRED:
            base.fail(f"Unexpected GT category: {category}")
        name = base.checked_filename(Path(row["Selected_Image"]).name, str(original))
        row["GT_Copy"] = str(gt_dir / category / (Path(name).stem + ".tif"))
    write_csv(output, list(rows[0]), rows)


def agreement_outputs(pred_rows: list[dict], gt_rows: list[dict],
                      args: argparse.Namespace, stage: Path) -> dict:
    import numpy as np
    import pandas as pd
    import section4_validation as validation

    selected = validation.read_selected(args.selection_dir)
    manual = validation.read_manual(args.manual_csv, selected)
    paired = selected[["Filename", "shape", "patient_id"]].merge(
        manual, on="Filename", how="left", validate="one_to_one")
    for source, rows, path in (
        ("pred", pred_rows, stage / "predicted_limb_144.csv"),
        ("gt", gt_rows, stage / "gt_limb_144.csv"),
    ):
        limb = validation.read_limb(path, selected)
        branch1 = pd.to_numeric(limb["Branch1_Diameter_px"], errors="coerce")
        branch2 = pd.to_numeric(limb["Branch2_Diameter_px"], errors="coerce")
        measured_pair = (np.isfinite(branch1) & np.isfinite(branch2) &
                         (branch1 > 0) & (branch2 > 0))
        # read_limb intentionally blanks Uncertain widths for its original
        # Success-only analysis. Recompute the width-ranked pair from the raw
        # branch values so the primary analysis retains measured QC cases.
        limb[f"{source}_thin_px"] = branch1.where(branch1 <= branch2, branch2).where(measured_pair)
        limb[f"{source}_wide_px"] = branch1.where(branch1 >= branch2, branch2).where(measured_pair)
        limb[f"{source}_measured_pair"] = measured_pair
        limb = limb.rename(columns={
            "limb_status": f"{source}_status",
            "limb_reason": f"{source}_reason",
            "Branch1_Diameter_px": f"{source}_branch1_px",
            "Branch2_Diameter_px": f"{source}_branch2_px",
            "limb_fold": f"{source}_fold",
        })[["Filename", f"{source}_status", f"{source}_reason",
            f"{source}_branch1_px", f"{source}_branch2_px",
            f"{source}_thin_px", f"{source}_wide_px",
            f"{source}_measured_pair", f"{source}_fold"]]
        loops = pd.DataFrame([{
            "Filename": row["Filename"],
            f"{source}_loop_px": float(row["Loop_Diameter_px"]),
        } for row in rows])
        paired = paired.merge(limb, on="Filename", how="left", validate="one_to_one")
        paired = paired.merge(loops, on="Filename", how="left", validate="one_to_one")
    if not (paired["pred_fold"] == paired["gt_fold"]).all():
        base.fail("GT and predicted exports do not use the same OOF fold")
    if not np.allclose(paired["pred_loop_px"], paired["gt_loop_px"], atol=0.01):
        base.fail("Loop diameter differs between mask arms despite common predicted U/D")
    if (len(paired) != 144 or not paired["Filename"].is_unique or
            set(paired["Filename"]) != {row["Filename"] for row in pred_rows}):
        base.fail("Agreement table does not match the selected image set")
    paired["common_measured"] = (paired["status"].eq("approved") &
                                  paired["pred_measured_pair"] &
                                  paired["gt_measured_pair"])
    paired["common_success"] = (paired["common_measured"] &
                                 paired["pred_status"].eq("Success") &
                                 paired["gt_status"].eq("Success"))

    def manual_summary(success_only: bool) -> pd.DataFrame:
        summary = []
        measures = (("thin_branch", "manual_thin_px"),
                    ("wide_branch", "manual_wide_px"))
        if not success_only:
            measures = (("loop", "manual_loop_px"), *measures)
        for source in ("pred", "gt"):
            for shape in ("all", "u", "tortuous", "twist"):
                part = paired if shape == "all" else paired.loc[paired["shape"] == shape]
                for measure, manual_col in measures:
                    auto_col = f"{source}_{'loop' if measure == 'loop' else measure.split('_')[0]}_px"
                    assessed = part.copy() if success_only else part
                    if success_only:
                        assessed[auto_col] = assessed[auto_col].where(
                            assessed[f"{source}_status"].eq("Success"))
                    result = validation.summary_row(
                        assessed, measure, "crossing" if shape == "twist" else shape,
                        manual_col, auto_col, args.bootstrap_reps,
                    )
                    result["analysis_set"] = "success_only" if success_only else "all_measured"
                    result["inclusion_rule"] = (
                        "approved + Success + two finite positive branch widths"
                        if success_only else
                        "approved + finite predicted U/D" if measure == "loop" else
                        "approved + two finite positive branch widths, any QC status"
                    )
                    result["mask_source"] = source
                    result["measured_pair_n"] = int(part[f"{source}_measured_pair"].sum())
                    result["success_n"] = int(part[f"{source}_status"].eq("Success").sum())
                    result["uncertain_n"] = int(part[f"{source}_status"].eq("Uncertain").sum())
                    result["failed_n"] = int(part[f"{source}_status"].eq("Failed").sum())
                    summary.append(result)
        return pd.DataFrame(summary)

    manual_summary(False).to_csv(stage / "manual_agreement_summary.csv",
                                 index=False, encoding="utf-8-sig")
    manual_summary(True).to_csv(stage / "manual_agreement_success_only.csv",
                                index=False, encoding="utf-8-sig")
    paired.to_csv(stage / "manual_agreement_by_image.csv",
                  index=False, encoding="utf-8-sig")

    def paired_comparison(success_only: bool) -> pd.DataFrame:
        common_rows = []
        for shape in ("all", "u", "tortuous", "twist"):
            group = paired if shape == "all" else paired.loc[paired["shape"] == shape]
            inclusion = "common_success" if success_only else "common_measured"
            part = group.loc[group[inclusion]]
            for measure in ("thin", "wide"):
                n = len(part)
                result = {
                    "analysis_set": "success_only" if success_only else "all_measured",
                    "shape": "crossing" if shape == "twist" else shape,
                    "measure": f"{measure}_branch", "paired_n": n,
                    "common_measured_n": int(group["common_measured"].sum()),
                    "common_success_n": int(group["common_success"].sum()),
                    "patient_n": int(part["patient_id"].nunique()),
                }
                if n:
                    pred = part[f"pred_{measure}_px"].to_numpy(dtype=float)
                    gt = part[f"gt_{measure}_px"].to_numpy(dtype=float)
                    manual_width = part[f"manual_{measure}_px"].to_numpy(dtype=float)
                    result.update({
                        "pred_mae_px": float(np.mean(np.abs(pred - manual_width))),
                        "gt_mae_px": float(np.mean(np.abs(gt - manual_width))),
                        "mae_delta_pred_minus_gt_px": float(np.mean(
                            np.abs(pred - manual_width) - np.abs(gt - manual_width))),
                        "pred_minus_gt_bias_px": float(np.mean(pred - gt)),
                    })
                common_rows.append(result)
        return pd.DataFrame(common_rows)

    paired_comparison(False).to_csv(stage / "paired_mask_comparison.csv",
                                    index=False, encoding="utf-8-sig")
    paired_comparison(True).to_csv(stage / "paired_mask_comparison_success_only.csv",
                                   index=False, encoding="utf-8-sig")
    return {
        "approved_n": int((paired["status"] == "approved").sum()),
        "pred_measured_pair_n": int(paired["pred_measured_pair"].sum()),
        "gt_measured_pair_n": int(paired["gt_measured_pair"].sum()),
        "pred_success_n": int((paired["pred_status"] == "Success").sum()),
        "gt_success_n": int((paired["gt_status"] == "Success").sum()),
        "common_measured_n": int(paired["common_measured"].sum()),
        "common_success_n": int(paired["common_success"].sum()),
    }


def execute(args: argparse.Namespace) -> None:
    selected, grouped, run_id = prepare_inputs(args)
    if args.output_dir.exists() and not args.dry_run:
        base.fail(f"Output already exists; choose a fresh --output-dir: {args.output_dir}")
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="resnet50_gt_manifest_") as temp:
            rebased = Path(temp) / "gt_mask_manifest_rebased.csv"
            rebase_gt_manifest(args.gt_dir, rebased)
            matches = gt_export.verify_gt_matches(
                selected, grouped, args.selection_dir, args.gt_dir, rebased)
        print(f"Dry run passed: {len(matches)} GT/predicted pairs, 5 OOF folds, run {run_id}")
        print("Category map: U=U, tortuous=tortuous, twist=crossing")
        return

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="section4_resnet50_stage_",
                                     dir=args.output_dir.parent) as temp:
        stage = Path(temp)
        rebased = stage / "gt_mask_manifest_rebased.csv"
        rebase_gt_manifest(args.gt_dir, rebased)
        matches = gt_export.verify_gt_matches(
            selected, grouped, args.selection_dir, args.gt_dir, rebased)
        cv2_site = gt_export.CV2_SITE
        if cv2_site.is_dir() and str(cv2_site) not in sys.path:
            sys.path.append(str(cv2_site))
        pred_rows = base.export_rows(grouped, run_id, LIMB_ROOT / "limb")
        gt_rows = gt_export.export_gt_rows(matches, run_id, LIMB_ROOT / "limb")
        if len(pred_rows) != 144 or len(gt_rows) != 144:
            base.fail("Exports must contain 144 rows per mask source")
        write_csv(stage / "predicted_limb_144.csv", base.FIELDNAMES, pred_rows)
        write_csv(stage / "gt_limb_144.csv", gt_export.FIELDNAMES, gt_rows)
        agreement = agreement_outputs(pred_rows, gt_rows, args, stage)
        details = {
            "run_id": run_id,
            "run_dir": str(args.run_dir),
            "selection_dir": str(args.selection_dir),
            "predicted_mask_dir": str(args.pred_dir),
            "gt_mask_dir": str(args.gt_dir),
            "gt_manifest_original": str(args.gt_dir / "gt_mask_manifest_144.csv"),
            "manual_csv": str(args.manual_csv),
            "category_map_gt_to_prediction": CATEGORY_TO_PRED,
            "selected_n": len(selected),
            "fold_counts": {str(fold): sum(item.fold == fold for item in grouped)
                            for fold in range(1, 6)},
            "pred_status_counts": {status: sum(row["Status"] == status for row in pred_rows)
                                   for status in ("Success", "Uncertain", "Failed")},
            "gt_status_counts": {status: sum(row["Status"] == status for row in gt_rows)
                                 for status in ("Success", "Uncertain", "Failed")},
            "agreement": agreement,
            "primary_inclusion": "Approved manual review and two finite positive measured branch widths, regardless of QC status",
            "sensitivity_inclusion": "Approved manual review and Status=Success in the relevant mask arm; paired comparison requires Success in both arms",
            "note": "Thin/wide are width-ranked branches, not confirmed physiological arteriole/venule.",
        }
        (stage / "run_metadata.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(args.output_dir)
    print(f"Saved {len(selected)} paired ROI measurements: {args.output_dir}")
    print(f"Measured branch pairs: predicted={agreement['pred_measured_pair_n']}, "
          f"GT={agreement['gt_measured_pair_n']}, "
          f"common approved={agreement['common_measured_n']}")
    print(f"Success-only sensitivity: predicted={agreement['pred_success_n']}, "
          f"GT={agreement['gt_success_n']}, "
          f"common approved={agreement['common_success_n']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--selection-dir", type=Path, default=SELECTION_DIR)
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--gt-dir", type=Path, default=GT_DIR)
    parser.add_argument("--manual-csv", type=Path, default=MANUAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be nonnegative")
    execute(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
