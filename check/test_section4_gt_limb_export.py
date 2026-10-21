"""Focused checks for Section 4 GT mask provenance and paired export schema."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import section4_gt_limb_export as gt


def write_manifest(path: Path, row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


class GTExportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.selection_dir = root / "selected"
        self.gt_dir = root / "GT"
        original_dir = root / "original"
        self.name = "p1_det_001.png"
        category = "U"
        self.selected_image = self.selection_dir / category / self.name
        self.selected_image.parent.mkdir(parents=True)
        self.selected_image.write_bytes(b"original image")
        self.gt_copy = self.gt_dir / category / "p1_det_001.tif"
        self.gt_copy.parent.mkdir(parents=True)
        original_dir.mkdir()
        self.gt_original = original_dir / self.gt_copy.name
        Image.new("L", (32, 32), color=255).save(self.gt_copy)
        self.gt_original.write_bytes(self.gt_copy.read_bytes())
        self.predicted_mask = root / "predicted" / self.name
        self.predicted_mask.parent.mkdir()
        self.predicted_mask.write_bytes(b"predicted mask")
        self.selected = [gt.base.SelectedImage(self.name, category, "p1")]
        self.prepared = [gt.base.PreparedImage(
            self.selected[0], 2,
            {"pred_U_x": "8", "pred_U_y": "8", "pred_D_x": "8",
             "pred_D_y": "16", "pred_length_px": "8"},
            self.predicted_mask, 32, 32)]
        self.manifest_path = root / "manifest.csv"
        sha = gt.base.file_sha256(self.gt_copy)
        self.manifest_row = {
            "Category": category,
            "Selected_Image": str(self.selected_image),
            "GT_Original": str(self.gt_original),
            "GT_Copy": str(self.gt_copy),
            "Original_SHA256": sha.upper(),
            "Copy_SHA256": sha.upper(),
            "Bytes": str(self.gt_copy.stat().st_size),
        }
        write_manifest(self.manifest_path, self.manifest_row)

    def verify(self):
        return gt.verify_gt_matches(self.selected, self.prepared,
                                    self.selection_dir, self.gt_dir,
                                    self.manifest_path)

    def test_gt_copy_only_replaces_mask_input(self) -> None:
        before_gt = self.gt_copy.read_bytes()
        before_original = self.gt_original.read_bytes()
        matches = self.verify()
        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match.gt_prepared.coordinates,
                         match.predicted.coordinates)
        self.assertEqual(match.gt_prepared.fold, match.predicted.fold)
        self.assertEqual(match.gt_prepared.mask_path, self.gt_copy)
        self.assertEqual(match.predicted.mask_path, self.predicted_mask)

        def fake_export(prepared, run_id, analyzer_dir):
            self.assertEqual([x.mask_path for x in prepared], [self.gt_copy])
            self.assertEqual(prepared[0].coordinates,
                             self.prepared[0].coordinates)
            return [{"Filename": self.name, "Mask_SHA256": match.gt_hash,
                     "Predicted_Mask": str(self.gt_copy),
                     "Algorithm_SHA256": "same-geometry-hash", "Status": "Uncertain",
                     "Branch1_Diameter_px": 8, "Branch2_Diameter_px": 12}]

        with patch.object(gt.base, "export_rows", side_effect=fake_export):
            rows = gt.export_gt_rows(matches, "run", self.gt_dir)
        self.assertEqual(rows[0]["Mask_Source"], "GT_Copy")
        self.assertEqual(rows[0]["Predicted_Mask"], str(self.predicted_mask))
        self.assertEqual(rows[0]["GT_Mask"], str(self.gt_copy))
        self.assertEqual(rows[0]["Mask_SHA256"], rows[0]["GT_Mask_SHA256"])
        self.assertEqual(rows[0]["Predicted_Mask_SHA256"],
                         gt.base.file_sha256(self.predicted_mask))
        self.assertEqual(rows[0]["Algorithm_SHA256"], "same-geometry-hash")
        self.assertEqual(rows[0]["Status"], "Uncertain")
        self.assertEqual(self.gt_copy.read_bytes(), before_gt)
        self.assertEqual(self.gt_original.read_bytes(), before_original)

    def test_wrong_gt_category_is_rejected(self) -> None:
        self.manifest_row["Category"] = "twist"
        write_manifest(self.manifest_path, self.manifest_row)
        with self.assertRaisesRegex(ValueError, "category mismatch"):
            self.verify()

    def test_copy_changed_since_manifest_is_rejected(self) -> None:
        Image.new("L", (32, 32), color=0).save(self.gt_copy)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.verify()

    def test_grayscale_equivalent_rgb_tiff_is_accepted(self) -> None:
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(self.gt_copy)
        self.gt_original.write_bytes(self.gt_copy.read_bytes())
        sha = gt.base.file_sha256(self.gt_copy).upper()
        self.manifest_row.update(Original_SHA256=sha, Copy_SHA256=sha,
                                 Bytes=str(self.gt_copy.stat().st_size))
        write_manifest(self.manifest_path, self.manifest_row)
        self.assertEqual(len(self.verify()), 1)

    def test_color_tiff_is_rejected(self) -> None:
        Image.new("RGB", (32, 32), color=(255, 0, 0)).save(self.gt_copy)
        self.gt_original.write_bytes(self.gt_copy.read_bytes())
        sha = gt.base.file_sha256(self.gt_copy).upper()
        self.manifest_row.update(Original_SHA256=sha, Copy_SHA256=sha,
                                 Bytes=str(self.gt_copy.stat().st_size))
        write_manifest(self.manifest_path, self.manifest_row)
        with self.assertRaisesRegex(ValueError, "unequal channels"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
