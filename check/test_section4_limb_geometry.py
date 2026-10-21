"""Regression checks for separated, measurable capillary limbs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


HERE = Path(__file__).resolve().parent
LIMB_DIR = HERE / "limb" if (HERE / "limb").is_dir() else HERE.parent / "limb"
sys.path.insert(0, str(LIMB_DIR))
# The existing OpenCV build is shared read-only with the Anaconda interpreter.
sys.path.append(str(Path.home() / ".conda" / "envs" / "younguk" / "Lib" / "site-packages"))

from analyzer import analyze_single_image, separated_limb_candidates  # noqa: E402
from core_algorithm import choose_best_endpoint_for_leg, find_endpoints  # noqa: E402


RUN = Path.home() / "Desktop" / "Multi-Task-Learning" / "results_unet_mtl_heatmap_bce_dice_5849db92da83"


class SeparatedLimbTests(unittest.TestCase):
    def test_background_gap_rejects_fused_bridge(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        mask[10:70, 16:25] = True
        mask[10:70, 46:55] = True
        mask[38:43, 24:47] = True
        left = [(y, 20) for y in range(12, 65)]
        right = [(y, 50) for y in range(12, 65)]
        valid = separated_limb_candidates(left, right, mask, np.array([0, 1]),
                                          distance_transform_edt(mask))
        self.assertTrue(valid[25 - 12])
        self.assertFalse(valid[40 - 12])
        self.assertTrue(valid[55 - 12])

    def test_distal_points_without_second_limb_are_rejected(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        mask[10:70, 16:25] = True
        mask[10:40, 46:55] = True
        long_path = [(y, 20) for y in range(12, 65)]
        short_path = [(y, 50) for y in range(12, 38)]
        valid = separated_limb_candidates(long_path, short_path, mask,
                                          np.array([0, 1]), distance_transform_edt(mask))
        self.assertTrue(valid[25 - 12])
        self.assertFalse(valid[55 - 12])

    def test_distal_cycle_without_endpoint_has_a_path(self) -> None:
        skeleton = np.zeros((35, 35), dtype=bool)
        skeleton[5:26, 10] = True
        skeleton[20:26, 15] = True
        skeleton[20, 10:16] = True
        skeleton[25, 10:16] = True
        seed = (11, 10)
        self.assertEqual(find_endpoints(skeleton), [(5, 10)])
        endpoint, path, _, _ = choose_best_endpoint_for_leg(
            skeleton, seed, (10.0, 12.0), np.array([0.0, 1.0]))
        self.assertIsNotNone(endpoint)
        self.assertGreater(endpoint[0], 20)
        self.assertGreater(len(path), 10)


class RealMaskRegressionTests(unittest.TestCase):
    @staticmethod
    def analyze(name: str, fold: int) -> dict:
        folder = RUN / f"fold_{fold}"
        table = pd.read_csv(folder / "predicted_coordinates.csv")
        row = table.loc[table.filename == name].iloc[0]
        points = pd.DataFrame([{
            "filename": name,
            "U_x": row.pred_U_x, "U_y": row.pred_U_y,
            "D_x": row.pred_D_x, "D_y": row.pred_D_y,
        }])
        return analyze_single_image(str(folder / "predicted_masks" / name), points)

    def test_fused_region_is_not_selected_as_branch_width(self) -> None:
        result = self.analyze("p17_det_007.png", 3)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["leg_results"]), 2)
        self.assertTrue(all(leg["max_y"] < 220 for leg in result["leg_results"]))

    def test_distal_bulb_is_not_selected(self) -> None:
        result = self.analyze("p1_det_031.png", 1)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["leg_results"]), 2)
        self.assertTrue(all(leg["max_y"] < 340 for leg in result["leg_results"]))

    def test_cycle_ended_branch_gets_a_distal_path(self) -> None:
        result = self.analyze("p4_det_018_crop_2.png", 1)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["leg_results"]), 2)


if __name__ == "__main__":
    unittest.main()
