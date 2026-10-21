"""Synthetic integrity and paired-denominator tests for mask-input comparison."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from section4_mask_paired_compare import compare  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PairedComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.selection = self.root / "selected"
        for shape in ("u", "tortuous", "twist"):
            (self.selection / shape).mkdir(parents=True)
        self.run = self.root / "run"
        fold = self.run / "fold_1"
        (fold / "predicted_masks").mkdir(parents=True)
        (self.run / "experiment_manifest.json").write_text('{"run_id": "test-run"}')
        self.gt_dir = self.root / "gt"
        self.gt_dir.mkdir()
        specifications = (("p1_a.png", "u", 10, 20, "Success"),
                          ("p1_b.png", "u", 12, 23, "Uncertain"),
                          ("p2_a.png", "tortuous", 14, 26, "Success"),
                          ("p3_a.png", "twist", 16, 29, "Success"))
        split_rows, coord_rows, manual_rows, manifest_rows = [], [], [], []
        pred_rows, gt_rows = [], []
        for index, (name, shape, thin, wide, pred_status) in enumerate(specifications):
            patient = name.split("_")[0]
            image = self.selection / shape / name
            image.write_bytes(f"image-{name}".encode())
            pred_mask = fold / "predicted_masks" / name
            pred_mask.write_bytes(f"pred-{name}".encode())
            gt_mask = self.gt_dir / f"{Path(name).stem}.tif"
            gt_mask.write_bytes(f"gt-{name}".encode())
            split_rows.append({"filename": name, "patient_id": patient,
                               "split": "outer_test"})
            coord_rows.append({"filename": name, "fold": 1,
                               "pred_U_x": 0, "pred_U_y": 0,
                               "pred_D_x": 0, "pred_D_y": 8 + index,
                               "pred_length_px": 8 + index})
            manual_rows.append({"Filename": name, "status": "approved",
                                "patient_id": patient, "shape": shape,
                                "B1_x1": 0, "B1_y1": 0, "B1_x2": thin, "B1_y2": 0,
                                "B2_x1": 0, "B2_y1": 0, "B2_x2": wide, "B2_y2": 0,
                                "U_x": 0, "U_y": 0, "D_x": 0, "D_y": 8 + index,
                                "manual_loop_px": 8 + index,
                                "manual_thin_px": thin, "manual_wide_px": wide})
            manifest_rows.append({"Category": shape, "Selected_Image": str(image),
                                  "GT_Original": str(self.root / "source" / gt_mask.name),
                                  "GT_Copy": str(gt_mask),
                                  "Original_SHA256": digest(gt_mask),
                                  "Copy_SHA256": digest(gt_mask)})
            common = {"Filename": name, "category": shape, "patient_id": patient,
                      "fold": 1, "Run_ID": "test-run", "Algorithm_SHA256": "algorithm",
                      "Predicted_Mask": str(pred_mask),
                      "Pred_U_x": 0, "Pred_U_y": 0,
                      "Pred_D_x": 0, "Pred_D_y": 8 + index,
                      "Loop_Diameter_px": 8 + index}
            pred_rows.append({**common, "Status": pred_status,
                              "Reason": "low ray count" if pred_status == "Uncertain" else "",
                              "Mask_SHA256": digest(pred_mask),
                              "Branch1_Diameter_px": thin + 2,
                              "Branch2_Diameter_px": wide + 2})
            gt_rows.append({**common, "Status": "Success", "Reason": "",
                            "Mask_SHA256": digest(gt_mask),
                            "Mask_Source": "GT_Copy", "GT_Mask": str(gt_mask),
                            "GT_Mask_SHA256": digest(gt_mask),
                            "Predicted_Mask_SHA256": digest(pred_mask),
                            "GT_Original": str(self.root / "source" / gt_mask.name),
                            "GT_Original_SHA256": digest(gt_mask),
                            "Branch1_Diameter_px": thin + 1,
                            "Branch2_Diameter_px": wide + 1})
        pd.DataFrame(split_rows).to_csv(fold / "patient_split.csv", index=False)
        pd.DataFrame(coord_rows).to_csv(fold / "predicted_coordinates.csv", index=False)
        self.manual = self.root / "manual.csv"
        self.pred = self.root / "pred.csv"
        self.gt = self.root / "gt.csv"
        self.manifest = self.root / "gt_manifest.csv"
        pd.DataFrame(manual_rows).to_csv(self.manual, index=False)
        pd.DataFrame(pred_rows).to_csv(self.pred, index=False)
        pd.DataFrame(gt_rows).to_csv(self.gt, index=False)
        pd.DataFrame(manifest_rows).to_csv(self.manifest, index=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def compare(self):
        return compare(self.selection, self.manual, self.pred, self.gt,
                       self.manifest, self.run, expected_count=4, bootstrap_reps=0)

    def test_common_success_keeps_uncertain_in_coverage_only(self) -> None:
        pairs, summary, coverage, provenance = self.compare()
        self.assertEqual(len(pairs), 4)
        self.assertEqual(provenance["common_success_n"], 3)
        self.assertEqual(int(coverage.loc[(coverage["shape"] == "all") &
                                          (coverage["pred_status"] == "Uncertain"), "count"].iloc[0]), 1)
        uncertain = pairs.loc[pairs["pred_status"] == "Uncertain"].iloc[0]
        self.assertTrue(pd.isna(uncertain["pred_thin_px"]))
        self.assertEqual(uncertain["pred_reported_thin_px"], 14)
        thin = summary.loc[(summary["shape"] == "all") &
                           (summary["target"] == "thin")].iloc[0]
        self.assertEqual(thin["common_success_n"], 3)
        self.assertAlmostEqual(thin["pred_mae_px"], 2)
        self.assertAlmostEqual(thin["gt_mae_px"], 1)
        self.assertAlmostEqual(thin["mae_improvement_gt_px"], 1)
        self.assertEqual(thin["patient_n"], 3)

    def test_tampered_copied_gt_mask_is_rejected(self) -> None:
        (self.gt_dir / "p2_a.tif").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "GT copied mask hash mismatch"):
            self.compare()

    def test_changed_predicted_keypoint_is_rejected(self) -> None:
        table = pd.read_csv(self.gt)
        table.loc[table["Filename"] == "p1_a.png", "Pred_D_y"] += 1
        table.to_csv(self.gt, index=False)
        with self.assertRaisesRegex(ValueError, "Predicted U/D changed"):
            self.compare()

    def test_different_algorithm_hash_is_rejected(self) -> None:
        table = pd.read_csv(self.gt)
        table["Algorithm_SHA256"] = "changed"
        table.to_csv(self.gt, index=False)
        with self.assertRaisesRegex(ValueError, "different algorithm source hashes"):
            self.compare()

    def test_manual_line_and_recorded_diameter_must_agree(self) -> None:
        table = pd.read_csv(self.manual)
        table.loc[table["Filename"] == "p1_a.png", "manual_thin_px"] = 99
        table.to_csv(self.manual, index=False)
        with self.assertRaisesRegex(ValueError, "Stale manual manual_thin_px"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
