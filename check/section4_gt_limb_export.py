"""Measure selected Section 4 branches on copied GT masks using OOF U/D.

This paired sensitivity export changes only the mask presented to the same
limb analyzer. The outer-test predicted coordinates, run, fold, branch QC, and
diameter estimator are taken from section4_oof_limb_export.py. The copied GT
mask manifest and the original GT files are verified without modification.

The output is a separate CSV with every predicted-mask export column plus
explicit GT/source provenance. It must not be interpreted as an end-to-end
MTL result or as ground-truth branch diameter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import sys


TARGET_CHECK = Path.home() / "Desktop" / "Capillary_limb" / "check"
if str(TARGET_CHECK) not in sys.path:
    sys.path.insert(0, str(TARGET_CHECK))
import section4_oof_limb_export as base  # noqa: E402


DEFAULT_RUN = (Path.home() / "Desktop" / "Multi-Task-Learning" /
               "results_unet_mtl_heatmap_bce_dice_5849db92da83")
DEFAULT_SELECTION = Path.home() / "Desktop" / "Pa_capillary"
DEFAULT_GT_DIR = TARGET_CHECK / "section4_gt_masks_pa_144"
DEFAULT_MANIFEST = DEFAULT_GT_DIR / "gt_mask_manifest_144.csv"
DEFAULT_OUTPUT = TARGET_CHECK / "section4_gt_limb_pa_144_v1.csv"
CV2_SITE = Path.home() / ".conda" / "envs" / "younguk" / "Lib" / "site-packages"

EXTRA_FIELDS = ["Mask_Source", "GT_Mask", "GT_Mask_SHA256",
                "Predicted_Mask_SHA256", "GT_Original", "GT_Original_SHA256"]
FIELDNAMES = [*base.FIELDNAMES, *EXTRA_FIELDS]


@dataclass(frozen=True)
class GTMatch:
    predicted: base.PreparedImage
    gt_prepared: base.PreparedImage
    gt_mask: Path
    gt_hash: str
    predicted_hash: str
    gt_original: Path
    gt_original_hash: str


def _inside(child: Path, parent: Path) -> bool:
    return child.resolve().is_relative_to(parent.resolve())


def verify_gt_matches(
    selected: list[base.SelectedImage],
    prepared: list[base.PreparedImage],
    selection_dir: Path,
    gt_dir: Path,
    manifest_path: Path,
) -> list[GTMatch]:
    """Require a one-to-one, hash checked GT copy for each selected image."""
    if len(selected) != len(prepared):
        base.fail("Selected and prepared OOF image counts differ")
    manifest = base.csv_rows(manifest_path, (
        "Category", "Selected_Image", "GT_Original", "GT_Copy",
        "Original_SHA256", "Copy_SHA256", "Bytes"))
    by_name: dict[str, dict[str, str]] = {}
    for row in manifest:
        image_name = base.checked_filename(Path(row["Selected_Image"]).name,
                                           str(manifest_path))
        if image_name in by_name:
            base.fail(f"Duplicate GT manifest image: {image_name}")
        by_name[image_name] = row
    if set(by_name) != {item.filename for item in selected}:
        missing = sorted({item.filename for item in selected} - set(by_name))
        extra = sorted(set(by_name) - {item.filename for item in selected})
        base.fail(f"GT manifest differs from selection: missing={missing[:8]}, "
                  f"extra={extra[:8]}")

    selected_by_name = {item.filename: item for item in selected}
    matches: list[GTMatch] = []
    for item in prepared:
        image = selected_by_name[item.selected.filename]
        if item.selected != image:
            base.fail(f"OOF selection mismatch: {image.filename}")
        row = by_name[image.filename]
        if row["Category"] != image.category:
            base.fail(f"GT category mismatch: {image.filename}")
        expected_image = (selection_dir / image.category / image.filename).resolve()
        actual_image = Path(row["Selected_Image"]).resolve()
        if actual_image != expected_image or not expected_image.is_file():
            base.fail(f"GT manifest selected-image path mismatch: {image.filename}")
        expected_copy = (gt_dir / image.category /
                         (Path(image.filename).stem + ".tif")).resolve()
        gt_copy = Path(row["GT_Copy"]).resolve()
        if gt_copy != expected_copy or not _inside(gt_copy, gt_dir) or not gt_copy.is_file():
            base.fail(f"GT copy path mismatch or missing: {image.filename}")
        gt_original = Path(row["GT_Original"]).resolve()
        if not gt_original.is_file():
            base.fail(f"GT original missing: {gt_original}")
        copied_hash = base.file_sha256(gt_copy)
        original_hash = base.file_sha256(gt_original)
        manifest_copied_hash = row["Copy_SHA256"].strip().lower()
        manifest_original_hash = row["Original_SHA256"].strip().lower()
        if not (copied_hash == original_hash == manifest_copied_hash ==
                manifest_original_hash):
            base.fail(f"GT original/copy SHA-256 mismatch: {image.filename}")
        if gt_copy.stat().st_size != int(row["Bytes"]):
            base.fail(f"GT copy size mismatch: {image.filename}")

        # Pillow checks TIFF dimensions without invoking the limb analyzer.
        from PIL import Image, ImageChops
        with Image.open(gt_copy) as image_file:
            width, height = image_file.size
            if image_file.n_frames != 1:
                base.fail(f"GT mask has multiple frames: {gt_copy}")
            if image_file.mode == "RGB":
                red, green, blue = image_file.split()
                if (ImageChops.difference(red, green).getbbox() or
                        ImageChops.difference(red, blue).getbbox()):
                    base.fail(f"GT RGB mask has unequal channels: {gt_copy}")
            elif image_file.mode not in {"L", "1"}:
                base.fail(f"GT mask is not grayscale-equivalent: {gt_copy}")
        if (width, height) != (item.mask_width, item.mask_height):
            base.fail(f"GT/OOF mask frame mismatch for {image.filename}: "
                      f"GT={width}x{height}, OOF={item.mask_width}x{item.mask_height}")
        gt_prepared = replace(item, mask_path=gt_copy,
                              mask_width=width, mask_height=height)
        base.finite_coordinate_values(gt_prepared)
        matches.append(GTMatch(item, gt_prepared, gt_copy, copied_hash,
                               base.file_sha256(item.mask_path),
                               gt_original, original_hash))
    return matches


def export_gt_rows(matches: list[GTMatch], run_id: str,
                   analyzer_dir: Path) -> list[dict[str, object]]:
    """Run identical geometry/QC on GT inputs and append unambiguous provenance."""
    rows = base.export_rows([match.gt_prepared for match in matches],
                            run_id, analyzer_dir)
    if len(rows) != len(matches):
        base.fail("GT exporter row count differs from verified input count")
    for row, match in zip(rows, matches):
        if row["Filename"] != match.predicted.selected.filename:
            base.fail("GT exporter row order differs from OOF selection order")
        if row["Mask_SHA256"] != match.gt_hash:
            base.fail(f"GT input hash mismatch during export: {row['Filename']}")
        row.update({
            "Predicted_Mask": str(match.predicted.mask_path),
            "Mask_Source": "GT_Copy",
            "GT_Mask": str(match.gt_mask),
            "GT_Mask_SHA256": match.gt_hash,
            "Predicted_Mask_SHA256": match.predicted_hash,
            "GT_Original": str(match.gt_original),
            "GT_Original_SHA256": match.gt_original_hash,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--gt-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--analyzer-dir", type=Path, default=base.DEFAULT_ANALYZER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify selected/OOF/GT correspondence; write nothing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    selected = base.select_images(args.selection_dir, None, 144)
    prepared, run_id = base.verify_run(args.run_dir.resolve(), selected)
    matches = verify_gt_matches(selected, prepared, args.selection_dir.resolve(),
                                args.gt_dir.resolve(), args.gt_manifest.resolve())
    print(f"Verified {len(matches)} copied GT masks against original hashes, "
          "selected images, OOF folds and predicted U/D coordinates")
    if args.dry_run:
        print("Dry run complete; no analyzer imported and no file written")
        return 0

    output = args.output.resolve()
    if output.suffix.lower() != ".csv" or not output.name.startswith("section4_gt_limb_"):
        parser.error("Output must be a distinct section4_gt_limb_*.csv")
    if not _inside(output, TARGET_CHECK) and not _inside(output, Path.cwd()):
        parser.error("Output must be inside Capillary_limb/check or the current workspace")
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists; pass --overwrite to replace it: {output}")
    if not CV2_SITE.is_dir():
        parser.error(f"Existing OpenCV package directory is missing: {CV2_SITE}")
    sys.path.append(str(CV2_SITE))
    import cv2  # noqa: F401
    import skimage  # noqa: F401
    rows = export_gt_rows(matches, run_id, args.analyzer_dir.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(row["Status"] == status for row in rows)
              for status in ("Success", "Uncertain", "Failed")}
    print(f"Exported {len(rows)} GT-mask rows ({counts}): {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
