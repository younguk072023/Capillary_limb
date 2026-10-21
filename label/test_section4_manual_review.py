"""Headless checks for Section 4 review data integrity."""

import csv
import tempfile
import unittest
from pathlib import Path

from label.section4_manual_review import load_reviews, save_reviews


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class Section4ReviewTest(unittest.TestCase):
    def test_grouped_images_start_blank_and_resume_in_one_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "Pa_capillary"
            for shape, name in (("U", "p1_u.png"),
                                ("tortuous", "p2_t.png"),
                                ("twist", "p3_w.png")):
                group = images / shape
                group.mkdir(parents=True)
                (group / name).write_bytes(b"original PNG placeholder")
            output = root / "manual_150.csv"
            missing = root / "no_legacy_drafts.csv"
            reviews = load_reviews(images, missing, missing, images, output,
                                   independent=True)
            self.assertEqual(len(reviews), 3)
            self.assertEqual({item.shape for item in reviews},
                             {"u", "tortuous", "twist"})
            self.assertEqual({item.image_relpath for item in reviews},
                             {"U/p1_u.png", "tortuous/p2_t.png",
                              "twist/p3_w.png"})
            self.assertTrue(all(all(point is None for point in item.points.values())
                                for item in reviews))
            first = reviews[0]
            first.points.update({"U": (0, 0), "D": (0, 4),
                                 "B1_1": (1, 1), "B1_2": (4, 5),
                                 "B2_1": (1, 1), "B2_2": (7, 9)})
            first.status = "approved"
            save_reviews(reviews, output)
            resumed = load_reviews(images, missing, missing, images, output,
                                   independent=True)
            self.assertEqual(resumed[0].metrics()["manual_loop_px"], 4.0)
            self.assertEqual(resumed[0].metrics()["manual_thin_px"], 5.0)
            self.assertEqual(resumed[0].metrics()["manual_wide_px"], 10.0)
            self.assertEqual([item.status for item in resumed],
                             ["approved", "pending", "pending"])

            (images / "twist" / "p1_u.png").write_bytes(b"different image")
            with self.assertRaisesRegex(ValueError, "Duplicate image filename"):
                load_reviews(images, missing, missing, images, output,
                             independent=True)

    def test_pending_import_approval_resume_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            for name in ("p1_a.png", "p2_b.png"):
                (images / name).write_bytes(b"original PNG placeholder")
            shapes = root / "shapes" / "u"
            shapes.mkdir(parents=True)
            (shapes / "p1_a.png").write_bytes(b"same")
            branches = root / "passivity.csv"
            loops = root / "keypoints.csv"
            output = root / "out" / "manual_review.csv"
            write_csv(branches,
                      ("Filename", "A_x1", "A_y1", "A_x2", "A_y2",
                       "V_x1", "V_y1", "V_x2", "V_y2"),
                      [{"Filename": "p1_a.png", "A_x1": 0, "A_y1": 0,
                        "A_x2": 3, "A_y2": 4, "V_x1": 0, "V_y1": 0,
                        "V_x2": 6, "V_y2": 8},
                       {"Filename": "p2_b.png", "A_x1": 0, "A_y1": 0,
                        "A_x2": 3, "A_y2": 4, "V_x1": 0, "V_y1": 0,
                        "V_x2": 6, "V_y2": 8}])
            write_csv(loops, ("filename", "U_x", "U_y", "D_x", "D_y"),
                      [{"filename": name, "U_x": 0, "U_y": 0,
                        "D_x": 0, "D_y": 12}
                       for name in ("p1_a.png", "p2_b.png")])
            before = branches.read_bytes(), loops.read_bytes()

            reviews = load_reviews(images, branches, loops, root / "shapes", output)
            self.assertFalse(output.exists())
            self.assertEqual([item.status for item in reviews], ["pending", "pending"])
            self.assertEqual(reviews[0].shape, "u")
            self.assertEqual(reviews[0].points["B1_2"], (3.0, 4.0))
            blind = load_reviews(images, branches, loops, root / "shapes", output,
                                 blind_loop=True)
            self.assertIsNone(blind[0].points["U"])
            self.assertIsNone(blind[0].points["D"])
            self.assertEqual(blind[0].points["B1_2"], (3.0, 4.0))
            reviews[0].status = "approved"
            reviews[0].reviewer = "Reviewer 1"
            self.assertEqual(reviews[0].metrics()["thin_branch"], "B1")
            save_reviews(reviews, output)
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["manual_loop_px"], "12.0")
            self.assertEqual(rows[0]["manual_thin_px"], "5.0")
            self.assertEqual(rows[0]["manual_wide_px"], "10.0")
            self.assertEqual(rows[1]["manual_thin_px"], "")
            self.assertEqual(rows[1]["status"], "pending")
            self.assertEqual((branches.read_bytes(), loops.read_bytes()), before)

            resumed = load_reviews(images, branches, loops, root / "shapes", output)
            self.assertEqual(resumed[0].status, "approved")
            self.assertEqual(resumed[0].reviewer, "Reviewer 1")
            resumed_blind = load_reviews(images, branches, loops, root / "shapes",
                                         output, blind_loop=True)
            self.assertEqual(resumed_blind[0].points["D"], (0.0, 12.0))
            resumed[1].status = "unmeasurable"
            with self.assertRaisesRegex(ValueError, "requires a reason"):
                save_reviews(resumed, output)
            resumed[1].reason = "vessel overlap"
            save_reviews(resumed, output)
            self.assertEqual(load_reviews(images, branches, loops,
                                          root / "shapes", output)[1].status,
                             "unmeasurable")

    def test_incomplete_legacy_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            (images / "p1_a.png").write_bytes(b"original")
            branches = root / "passivity.csv"
            write_csv(branches, ("Filename", "A_x1", "A_y1"),
                      [{"Filename": "p1_a.png", "A_x1": 7, "A_y1": ""}])
            with self.assertRaisesRegex(ValueError, "Incomplete legacy point"):
                load_reviews(images, branches, root / "unused.csv", root / "shapes",
                             root / "review.csv", blind_loop=True)


if __name__ == "__main__":
    unittest.main()
