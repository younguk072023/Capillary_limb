"""Focused provenance and endpoint checks for the Section 4 OOF exporter."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from check.section4_oof_limb_export import (SelectedImage, branch_measurement,
                                            export_rows, verify_run)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class OOFExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        self.run.mkdir()
        self.names = ("p1_det_001.png", "p2_det_001.png")
        manifest = {"run_id": "synthetic", "run_spec": {
            "experiment": {"task_mode": "mtl"}, "n_splits": 2,
            "image_inventory": [[name, 100, 0] for name in self.names],
        }}
        (self.run / "experiment_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        self.write_fold(1, self.names[0])
        self.write_fold(2, self.names[1])
        self.selected = [SelectedImage(name, "u", name.split("_det")[0])
                         for name in self.names]

    def write_fold(self, number: int, test_name: str) -> None:
        folder = self.run / f"fold_{number}"
        folder.mkdir(exist_ok=True)
        (folder / "fold_summary.txt").write_text("complete", encoding="utf-8")
        write_csv(folder / "patient_split.csv", ["filename", "patient_id", "split"],
                  [{"filename": name, "patient_id": name.split("_det")[0],
                    "split": "outer_test" if name == test_name else "train"}
                   for name in self.names])
        write_csv(folder / "evaluated_images.csv", ["filename"],
                  [{"filename": test_name}])
        write_csv(folder / "predicted_coordinates.csv",
                  ["filename", "fold", "pred_U_x", "pred_U_y",
                   "pred_D_x", "pred_D_y", "pred_length_px"],
                  [{"filename": test_name, "fold": number,
                    "pred_U_x": 10, "pred_U_y": 10,
                    "pred_D_x": 10, "pred_D_y": 20,
                    "pred_length_px": 10}])
        masks = folder / "predicted_masks"
        masks.mkdir(exist_ok=True)
        for old in masks.glob("*.png"):
            old.unlink()
        # IHDR header suffices for the dependency-free provenance check.
        (masks / test_name).write_bytes(
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" +
            struct.pack(">II", 32, 32))

    def test_selected_images_have_unique_outer_test_artifacts(self) -> None:
        prepared, run_id = verify_run(self.run, self.selected)
        self.assertEqual(run_id, "synthetic")
        self.assertEqual([(x.selected.filename, x.fold) for x in prepared],
                         [(self.names[0], 1), (self.names[1], 2)])

    def test_repeated_outer_test_is_rejected(self) -> None:
        self.write_fold(2, self.names[0])
        with self.assertRaisesRegex(ValueError, "not one-time outer-test OOF"):
            verify_run(self.run, self.selected)

    def test_endpoint_line_reproduces_analyzer_width(self) -> None:
        width, x1, y1, x2, y2 = branch_measurement({
            "max_d": 10, "max_x": 20, "max_y": 30,
            "local_perp_v": (1, 0), "max_d_pos": 4, "max_d_neg": 6,
        })
        self.assertEqual((width, x1, y1, x2, y2), (10, 24, 30, 14, 30))

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas unavailable")
    def test_export_keeps_physical_branch_order_and_size_labels(self) -> None:
        prepared, _ = verify_run(self.run, self.selected)
        analyzer_dir = Path(self.temporary.name) / "analyzer"
        analyzer_dir.mkdir()
        for name in ("analyzer.py", "core_algorithm.py", "utils.py"):
            (analyzer_dir / name).write_text("# synthetic\n", encoding="utf-8")

        def fake_analyzer(mask_path, keypoints):
            self.assertEqual(keypoints.iloc[0]["filename"], Path(mask_path).name)
            return {"ok": True, "leg_results": [
                {"id": 0, "mean_x": 10, "max_d": 12,
                 "max_x": 10, "max_y": 20, "local_perp_v": (1, 0),
                 "max_d_pos": 6, "max_d_neg": 6,
                 "path_points": 30, "topology_eligible_points": 20,
                 "ray_valid_points": 12, "used_edt_fallback": False},
                {"id": 1, "mean_x": 22, "max_d": 8,
                 "max_x": 22, "max_y": 20, "local_perp_v": (1, 0),
                 "max_d_pos": 4, "max_d_neg": 4,
                 "path_points": 30, "topology_eligible_points": 18,
                 "ray_valid_points": 10, "used_edt_fallback": False},
            ]}

        with patch("check.section4_oof_limb_export.load_analyzer", return_value=fake_analyzer):
            rows = export_rows(prepared, "synthetic", analyzer_dir)
        self.assertEqual([row["Status"] for row in rows], ["Success", "Success"])
        self.assertEqual((rows[0]["Branch1_Diameter_px"],
                          rows[0]["Branch2_Diameter_px"]), (12, 8))
        self.assertEqual((rows[0]["Estimated_Arterial_Branch"],
                          rows[0]["Estimated_Venous_Branch"]), (2, 1))
        self.assertEqual((rows[0]["Branch1_x1"], rows[0]["Branch1_x2"]),
                         (16, 4))
        self.assertEqual((rows[0]["Branch1_Ray_Valid_Points"],
                          rows[0]["Branch2_Ray_Valid_Points"]), (12, 10))

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas unavailable")
    def test_few_valid_rays_are_exported_as_uncertain(self) -> None:
        prepared, _ = verify_run(self.run, self.selected)
        analyzer_dir = Path(self.temporary.name) / "analyzer"
        analyzer_dir.mkdir()
        for name in ("analyzer.py", "core_algorithm.py", "utils.py"):
            (analyzer_dir / name).write_text("# synthetic\n", encoding="utf-8")

        def fake_analyzer(mask_path, keypoints):
            return {"ok": True, "leg_results": [
                {"id": index, "mean_x": index * 20, "max_d": 8,
                 "max_x": index * 20, "max_y": 20,
                 "local_perp_v": (1, 0), "max_d_pos": 4, "max_d_neg": 4,
                 "path_points": 25, "topology_eligible_points": 10,
                 "ray_valid_points": 4 if index == 0 else 8,
                 "used_edt_fallback": False}
                for index in (0, 1)]}

        with patch("check.section4_oof_limb_export.load_analyzer", return_value=fake_analyzer):
            rows = export_rows(prepared, "synthetic", analyzer_dir)
        self.assertEqual([row["Status"] for row in rows], ["Uncertain", "Uncertain"])
        self.assertIn("only 4 ray-valid points", rows[0]["Reason"])
        self.assertEqual(rows[0]["Branch1_Diameter_px"], 8)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas unavailable")
    def test_severely_unbalanced_branch_paths_are_uncertain(self) -> None:
        prepared, _ = verify_run(self.run, self.selected)
        analyzer_dir = Path(self.temporary.name) / "analyzer"
        analyzer_dir.mkdir()
        for name in ("analyzer.py", "core_algorithm.py", "utils.py"):
            (analyzer_dir / name).write_text("# synthetic\n", encoding="utf-8")

        def fake_analyzer(mask_path, keypoints):
            return {"ok": True, "leg_results": [
                {"id": index, "mean_x": index * 20, "max_d": 8,
                 "max_x": index * 20, "max_y": 20,
                 "local_perp_v": (1, 0), "max_d_pos": 4, "max_d_neg": 4,
                 "path_points": 20 if index == 0 else 100,
                 "topology_eligible_points": 10,
                 "ray_valid_points": 8, "used_edt_fallback": False}
                for index in (0, 1)]}

        with patch("check.section4_oof_limb_export.load_analyzer", return_value=fake_analyzer):
            rows = export_rows(prepared, "synthetic", analyzer_dir)
        self.assertEqual([row["Status"] for row in rows], ["Uncertain", "Uncertain"])
        self.assertIn("path imbalance", rows[0]["Reason"])
        self.assertAlmostEqual(rows[0]["Branch_Path_Balance"], 0.2)


if __name__ == "__main__":
    unittest.main()
