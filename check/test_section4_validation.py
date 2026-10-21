"""Data-integrity checks for the Section 4 evaluation join."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from check.section4_validation import icc_pair, read_limb, read_manual, read_oof


class Section4ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.selected = pd.DataFrame([{"Filename": "p1_a.png", "shape": "u", "patient_id": "p1"}])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_absolute_icc_detects_fixed_bias_that_consistency_ignores(self) -> None:
        a = np.array([10, 20, 30, 40], dtype=float)
        agreement, consistency = icc_pair(a, a + 5)
        self.assertAlmostEqual(consistency, 1.0)
        self.assertLess(agreement, consistency)

    def test_approved_manual_requires_both_independent_lines(self) -> None:
        file = self.root / "manual.csv"
        row = {"Filename": "p1_a.png", "status": "approved",
               "B1_x1": 0, "B1_y1": 0, "B1_x2": 3, "B1_y2": 4,
               "B2_x1": 0, "B2_y1": 0, "B2_x2": np.nan, "B2_y2": 8,
               "U_x": 0, "U_y": 0, "D_x": 0, "D_y": 12,
               "manual_loop_px": 12, "manual_thin_px": 5, "manual_wide_px": 10}
        pd.DataFrame([row]).to_csv(file, index=False)
        with self.assertRaisesRegex(ValueError, "Non-finite endpoints"):
            read_manual(file, self.selected)

    def test_failed_limb_does_not_reuse_partial_width(self) -> None:
        file = self.root / "limb.csv"
        pd.DataFrame([{"Filename": "p1_a.png", "fold": 1, "Status": "Failed",
                       "Reason": "no second branch", "Branch1_Diameter_px": 20,
                       "Branch2_Diameter_px": np.nan, "Run_ID": "example",
                       "Predicted_Mask": "mask.png", "Mask_SHA256": "hash",
                       "Algorithm_SHA256": "algorithm"}]).to_csv(file, index=False)
        result = read_limb(file, self.selected)
        self.assertTrue(np.isnan(result.at[0, "pred_thin_px"]))
        self.assertTrue(np.isnan(result.at[0, "pred_wide_px"]))

    def test_oof_fold_mismatch_is_rejected(self) -> None:
        fold = self.root / "fold_1"
        (fold / "predicted_masks").mkdir(parents=True)
        (fold / "predicted_masks" / "p1_a.png").write_bytes(b"mask")
        pd.DataFrame([{"filename": "p1_a.png", "patient_id": "p1",
                       "split": "outer_test"}]).to_csv(fold / "patient_split.csv", index=False)
        pd.DataFrame([{"filename": "p1_a.png", "fold": 2,
                       "pred_U_x": 0, "pred_U_y": 0, "pred_D_x": 0,
                       "pred_D_y": 10, "pred_length_px": 10}]).to_csv(
                           fold / "predicted_coordinates.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Coordinate fold mismatch"):
            read_oof(self.root, self.selected)


if __name__ == "__main__":
    unittest.main()
