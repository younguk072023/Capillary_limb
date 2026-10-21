"""Compare original-image manual diameters with GT-mask + MTL-predicted-U/D geometry.

Use a completed patient-out-of-fold MTL run and its separate GT-mask analyzer
export. The GT segmentation copy is the analyzer mask input; U/D remains the
MTL prediction. This is a mask-input sensitivity result, not the end-to-end
MTL result. Loop diameter uses U/D only and is independent of the mask input.

Run the matching GT-mask exporter first, then this script. Every selected image
remains in the per-image and coverage outputs. Branch agreement uses approved
manual reviews and analyzer Status=Success; Uncertain/Failed stay visible but
are excluded from branch agreement. Valid loop pairs are retained regardless
of branch status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

sys.dont_write_bytecode = True
TARGET_CHECK = Path.home() / "Desktop" / "Capillary_limb" / "check"
if str(TARGET_CHECK) not in sys.path:
    sys.path.append(str(TARGET_CHECK))

import section4_mask_paired_compare as provenance_check
import section4_validation as validation


DEFAULT_SELECTION = Path.home() / "Desktop" / "Pa_capillary"
DEFAULT_MANUAL = (Path.home() / "Desktop" / "Capillary_limb" / "label" /
                  "outputs" / "section4_review_pa_144" / "manual_review_144.csv")
DEFAULT_GT_MANIFEST = (Path.home() / "Desktop" / "Capillary_limb" / "check" /
                       "section4_gt_masks_pa_144" / "gt_mask_manifest_144.csv")
DEFAULT_OOF_RUN = (Path.home() / "Desktop" / "Multi-Task-Learning" /
                   "results_unet_mtl_heatmap_bce_dice_5849db92da83")
DEFAULT_GT_EXPORT = (Path.home() / "Desktop" / "Capillary_limb" / "check" /
                     "section4_gt_limb_pa_144_v1.csv")
DEFAULT_OUTPUT = (Path.home() / "Desktop" / "Capillary_limb" / "check" /
                  "section4_gt_predud_agreement_pa_144_v1")

OUTPUT_NAMES = {
    "pairs": "section4_gt_predud_pairs.csv",
    "agreement": "section4_gt_predud_agreement.csv",
    "coverage": "section4_gt_predud_coverage.csv",
    "provenance": "section4_gt_predud_provenance.json",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_original_gt_sources(manifest: pd.DataFrame) -> None:
    """The copied GT input must still equal its original label byte for byte."""
    for name, row in manifest.iterrows():
        original = Path(row["GT_Original"])
        if not original.is_file():
            raise FileNotFoundError(f"Original GT mask missing: {name}: {original}")
        if file_hash(original).lower() != str(row["Original_SHA256"]).lower():
            raise ValueError(f"Original GT mask hash changed: {name}")


def make_pairs(selection: pd.DataFrame, manual: pd.DataFrame,
               gt_export: pd.DataFrame) -> pd.DataFrame:
    """Retain all 144 rows; mark which values enter agreement statistics."""
    rows = []
    for item in selection.itertuples(index=False):
        name = item.Filename
        m, g = manual.loc[name], gt_export.loc[name]
        reported_thin, reported_wide = provenance_check._reported_widths(g)
        branch_pair = m["status"] == "approved" and g["Status"] == "Success"
        loop = float(g["Loop_Diameter_px"])
        valid_loop = (m["status"] == "approved" and math.isfinite(loop) and loop > 0)
        row = {
            "Filename": name,
            "shape": item.category,
            "patient_id": item.patient_id,
            "fold": int(g["fold"]),
            "manual_status": m["status"],
            "algorithm_status": g["Status"],
            "algorithm_reason": str(g["Reason"]) if pd.notna(g["Reason"]) else "",
            "loop_pair": valid_loop,
            "branch_pair": branch_pair,
            "manual_loop_px": float(m["manual_loop_px"]),
            "algorithm_loop_px": loop if valid_loop else math.nan,
            "manual_thin_px": float(m["manual_thin_px"]),
            "manual_wide_px": float(m["manual_wide_px"]),
            "gt_reported_thin_px": reported_thin,
            "gt_reported_wide_px": reported_wide,
            "algorithm_thin_px": reported_thin if branch_pair else math.nan,
            "algorithm_wide_px": reported_wide if branch_pair else math.nan,
            "gt_mask": str(g["GT_Mask"]),
            "predicted_mask_reference": str(g["Predicted_Mask"]),
            "algorithm_sha256": str(g["Algorithm_SHA256"]),
        }
        for target in ("loop", "thin", "wide"):
            delta = row[f"algorithm_{target}_px"] - row[f"manual_{target}_px"]
            row[f"error_{target}_px"] = delta
            row[f"abs_error_{target}_px"] = abs(delta)
        rows.append(row)
    return pd.DataFrame(rows)


def make_agreement(pairs: pd.DataFrame, bootstrap_reps: int) -> pd.DataFrame:
    """Pearson and two-way single-measure ICCs with patient-cluster CIs."""
    rows = []
    metric_table = pairs.copy()
    metric_table["status"] = metric_table["manual_status"]
    for shape in ("all", "u", "tortuous", "twist"):
        part = metric_table if shape == "all" else metric_table.loc[metric_table["shape"] == shape]
        for target in ("loop", "thin", "wide"):
            result = validation.summary_row(
                part, target, shape, f"manual_{target}_px",
                f"algorithm_{target}_px", bootstrap_reps)
            result.update({
                "mask_source": "GT_Copy",
                "keypoint_source": "MTL_Predicted_OOF",
                "algorithm_success_n": int(part["algorithm_status"].eq("Success").sum()),
                "algorithm_uncertain_n": int(part["algorithm_status"].eq("Uncertain").sum()),
                "algorithm_failed_n": int(part["algorithm_status"].eq("Failed").sum()),
            })
            rows.append(result)
    return pd.DataFrame(rows)


def make_coverage(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shape in ("all", "u", "tortuous", "twist"):
        part = pairs if shape == "all" else pairs.loc[pairs["shape"] == shape]
        for status in ("Success", "Uncertain", "Failed"):
            subset = part.loc[part["algorithm_status"] == status]
            rows.append({"shape": shape, "selected_n": len(part),
                         "algorithm_status": status, "image_n": len(subset),
                         "patient_n": int(subset["patient_id"].nunique())})
    return pd.DataFrame(rows)


def evaluate(selection_dir: Path, manual_csv: Path, gt_manifest_csv: Path,
             oof_run_dir: Path, gt_export_csv: Path,
             bootstrap_reps: int = 1000) -> tuple[dict[str, pd.DataFrame], dict]:
    selection = provenance_check.read_selection(selection_dir, expected_count=144)
    manual = provenance_check.read_manual(manual_csv, selection)
    if not manual["status"].eq("approved").all():
        raise ValueError("All 144 manual measurements must be approved")
    run_id, oof = provenance_check.read_oof(oof_run_dir, selection)
    gt_manifest = provenance_check.read_gt_manifest(gt_manifest_csv, selection)
    verify_original_gt_sources(gt_manifest)
    gt_export = provenance_check.read_export(
        gt_export_csv, selection, oof, run_id, gt_manifest)
    if not gt_export["Algorithm_SHA256"].nunique() == 1:
        raise ValueError("GT export has inconsistent algorithm source hashes")
    pairs = make_pairs(selection, manual, gt_export)
    if len(pairs) != 144 or not pairs["Filename"].is_unique:
        raise ValueError("Agreement join did not preserve 144 unique images")
    agreement = make_agreement(pairs, bootstrap_reps)
    coverage = make_coverage(pairs)
    metadata = {
        "design": "Copied GT segmentation mask + patient-out-of-fold MTL-predicted U/D",
        "interpretation": "Mask-input sensitivity; not the end-to-end MTL result",
        "loop_interpretation": "MTL predicted U/D versus independent manual U/D; mask input does not affect loop diameter",
        "branch_inclusion": "Manual approved and GT-mask analyzer Status=Success",
        "loop_inclusion": "Manual approved and valid predicted U/D, regardless of branch status",
        "selected_n": len(selection),
        "selected_patient_n": int(selection["patient_id"].nunique()),
        "loop_paired_n": int(pairs["loop_pair"].sum()),
        "branch_paired_n": int(pairs["branch_pair"].sum()),
        "bootstrap_reps": bootstrap_reps,
        "run_id": run_id,
        "algorithm_sha256": str(gt_export["Algorithm_SHA256"].iloc[0]),
        "selection_dir": str(selection_dir.resolve()),
        "manual_csv": str(manual_csv.resolve()),
        "manual_csv_sha256": file_hash(manual_csv),
        "gt_manifest_csv": str(gt_manifest_csv.resolve()),
        "gt_manifest_sha256": file_hash(gt_manifest_csv),
        "oof_run_dir": str(oof_run_dir.resolve()),
        "oof_run_manifest_sha256": file_hash(oof_run_dir / "experiment_manifest.json"),
        "gt_export_csv": str(gt_export_csv.resolve()),
        "gt_export_sha256": file_hash(gt_export_csv),
    }
    return {"pairs": pairs, "agreement": agreement, "coverage": coverage}, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--gt-manifest", type=Path, default=DEFAULT_GT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OOF_RUN)
    parser.add_argument("--gt-export-csv", type=Path, default=DEFAULT_GT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be nonnegative")
    existing = [args.output_dir / name for name in OUTPUT_NAMES.values()
                if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error(f"Output already exists; pass --overwrite to replace: {existing}")
    tables, metadata = evaluate(args.selection_dir, args.manual_csv,
                                args.gt_manifest, args.run_dir,
                                args.gt_export_csv, args.bootstrap_reps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        table.to_csv(args.output_dir / OUTPUT_NAMES[key], index=False,
                     encoding="utf-8-sig")
    (args.output_dir / OUTPUT_NAMES["provenance"]).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {metadata['selected_n']} images, "
          f"{metadata['loop_paired_n']} loop pairs, "
          f"{metadata['branch_paired_n']} Success branch pairs to "
          f"{args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
