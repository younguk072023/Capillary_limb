"""Export Section 4 limb diameters from a completed MTL run's outer-test masks.

The selected images are read from ``analysis/image/{u,tortuous,twist}`` by
default. No training, prediction, source, or mask file is changed. The
``--dry-run`` path verifies fold provenance using only the Python standard
library; the full export additionally needs the dependencies of
``Capillary_limb/limb/analyzer.py``.

Branch endpoint columns reproduce the analyzer's displayed measurement line.
The analyzer currently uses an approximate display line when it falls back to
EDT; the exported fallback flag marks these approximate endpoints. Branches
with fewer than five ray-valid candidate points or EDT fallback are marked
``Uncertain`` and excluded from the primary agreement calculation.

Example (PowerShell)::

    & $py check/section4_oof_limb_export.py --run-dir results_unet_mtl_heatmap_bce_dice_5849db92da83 --dry-run
    & $py check/section4_oof_limb_export.py --run-dir results_unet_mtl_heatmap_bce_dice_5849db92da83
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_DIR = Path.home() / "Desktop" / "analysis" / "image"
DEFAULT_ANALYZER_DIR = Path.home() / "Desktop" / "Capillary_limb" / "limb"
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
FOLD_PATTERN = re.compile(r"fold_(\d+)$")

FIELDNAMES = [
    "Filename", "category", "patient_id", "fold", "Status", "Reason",
    "Run_ID", "Algorithm_SHA256", "Predicted_Mask", "Mask_SHA256",
    "Pred_U_x", "Pred_U_y", "Pred_D_x", "Pred_D_y", "Loop_Diameter_px",
    "Branch1_ID", "Branch1_Side", "Branch1_Diameter_px",
    "Branch1_x1", "Branch1_y1", "Branch1_x2", "Branch1_y2",
    "Branch1_Path_Points", "Branch1_Topology_Candidates",
    "Branch1_Ray_Valid_Points", "Branch1_EDT_Fallback",
    "Branch2_ID", "Branch2_Side", "Branch2_Diameter_px",
    "Branch2_x1", "Branch2_y1", "Branch2_x2", "Branch2_y2",
    "Branch2_Path_Points", "Branch2_Topology_Candidates",
    "Branch2_Ray_Valid_Points", "Branch2_EDT_Fallback",
    "Branch_Path_Balance",
    "Estimated_Arterial_Diameter_px", "Estimated_Arterial_Branch",
    "Estimated_Venous_Diameter_px", "Estimated_Venous_Branch",
]

MIN_VALID_RAY_POINTS = 5
MIN_BRANCH_PATH_BALANCE = 1.0 / 3.0


@dataclass(frozen=True)
class SelectedImage:
    filename: str
    category: str
    patient_id: str


@dataclass(frozen=True)
class PreparedImage:
    selected: SelectedImage
    fold: int
    coordinates: dict[str, str]
    mask_path: Path
    mask_width: int
    mask_height: int


def fail(message: str) -> None:
    raise ValueError(message)


def csv_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Required CSV is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            fail(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def checked_filename(value: str, context: str) -> str:
    name = str(value or "").strip()
    if not name or Path(name).name != name or "\\" in name or "/" in name:
        fail(f"Invalid filename in {context}: {value!r}")
    return name


def patient_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_det" not in stem:
        fail(f"Cannot derive patient ID from filename: {filename}")
    return stem.split("_det", 1)[0]


def select_images(selection_dir: Path | None, selection_csv: Path | None,
                  expected_count: int) -> list[SelectedImage]:
    images: list[SelectedImage] = []
    if selection_csv is not None:
        rows = csv_rows(selection_csv, ())
        for row in rows:
            name = checked_filename(row.get("Filename") or row.get("filename"),
                                    str(selection_csv))
            category = (row.get("category") or row.get("Category") or "").strip()
            images.append(SelectedImage(name, category, patient_from_filename(name)))
    else:
        directory = selection_dir or DEFAULT_SELECTION_DIR
        if not directory.is_dir():
            fail(f"Selection directory does not exist: {directory}")
        for category_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
            for image in sorted(path for path in category_dir.iterdir()
                                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                name = checked_filename(image.name, str(category_dir))
                images.append(SelectedImage(name, category_dir.name,
                                            patient_from_filename(name)))
    if not images:
        fail("The selection contains no images")
    seen: set[str] = set()
    for item in images:
        key = item.filename.casefold()
        if key in seen:
            fail(f"Selected filename occurs more than once: {item.filename}")
        seen.add(key)
    if expected_count and len(images) != expected_count:
        fail(f"Expected {expected_count} selected images, found {len(images)}")
    return images


def index_unique(rows: list[dict[str, str]], column: str,
                 context: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = checked_filename(row.get(column), str(context))
        if name in result:
            fail(f"Duplicate {column}={name!r} in {context}")
        result[name] = row
    return result


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        fail(f"Predicted mask is not a valid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        fail(f"Predicted mask has invalid dimensions: {path}")
    return width, height


def verify_run(run_dir: Path, selected: list[SelectedImage]
               ) -> tuple[list[PreparedImage], str]:
    manifest_path = run_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        fail(f"Completed MTL run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    run_spec = manifest.get("run_spec", {})
    if run_spec.get("experiment", {}).get("task_mode") != "mtl":
        fail(f"Run is not an MTL experiment: {run_dir}")
    expected_folds = int(run_spec.get("n_splits", 0))
    if expected_folds < 2:
        fail(f"Run manifest has no valid n_splits: {manifest_path}")
    fold_dirs = sorted((path for path in run_dir.iterdir()
                        if path.is_dir() and FOLD_PATTERN.fullmatch(path.name)),
                       key=lambda path: int(FOLD_PATTERN.fullmatch(path.name).group(1)))
    fold_numbers = [int(FOLD_PATTERN.fullmatch(path.name).group(1)) for path in fold_dirs]
    if fold_numbers != list(range(1, expected_folds + 1)):
        fail(f"Expected completed fold_1..fold_{expected_folds}; found {fold_numbers}")

    chosen = {image.filename: image for image in selected}
    inventory = run_spec.get("image_inventory")
    if inventory is not None:
        inventory_names = {str(item[0]) for item in inventory}
        missing = set(chosen) - inventory_names
        if missing:
            fail(f"Selected images absent from run image inventory: {sorted(missing)[:8]}")

    outer_test_matches: dict[str, list[PreparedImage]] = {name: [] for name in chosen}
    all_test_counts: dict[str, int] = {}
    split_name_sets: list[set[str]] = []
    for fold, fold_dir in zip(fold_numbers, fold_dirs):
        if not (fold_dir / "fold_summary.txt").is_file():
            fail(f"Fold completion marker is missing: {fold_dir / 'fold_summary.txt'}")
        split_path = fold_dir / "patient_split.csv"
        split = index_unique(csv_rows(split_path, ("filename", "patient_id", "split")),
                             "filename", split_path)
        split_name_sets.append(set(split))
        patient_roles: dict[str, set[str]] = {}
        for name, row in split.items():
            patient = patient_from_filename(name)
            if row["patient_id"].strip() != patient:
                fail(f"Patient ID mismatch for {name} in {split_path}")
            role = row["split"].strip()
            if role not in {"train", "inner_val", "outer_test"}:
                fail(f"Unexpected split role {role!r} for {name} in {split_path}")
            patient_roles.setdefault(patient, set()).add(role)
        mixed = [patient for patient, roles in patient_roles.items() if len(roles) > 1]
        if mixed:
            fail(f"Patients span split roles in {split_path}: {mixed[:8]}")

        test_names = {name for name, row in split.items()
                      if row["split"].strip() == "outer_test"}
        evaluated_path = fold_dir / "evaluated_images.csv"
        evaluated = index_unique(csv_rows(evaluated_path, ("filename",)),
                                 "filename", evaluated_path)
        if set(evaluated) != test_names:
            fail(f"Evaluated image list differs from outer_test in {fold_dir}")
        coordinate_path = fold_dir / "predicted_coordinates.csv"
        coords = index_unique(csv_rows(coordinate_path,
                                       ("filename", "pred_U_x", "pred_U_y",
                                        "pred_D_x", "pred_D_y")),
                              "filename", coordinate_path)
        if set(coords) != test_names:
            missing = sorted(test_names - set(coords))[:8]
            extra = sorted(set(coords) - test_names)[:8]
            fail(f"Coordinates differ from outer_test in {fold_dir}: missing={missing}, extra={extra}")
        for name, coord in coords.items():
            if coord.get("fold") and float(coord["fold"]) != fold:
                fail(f"Coordinate fold mismatch for {name} in {coordinate_path}")

        mask_dir = fold_dir / "predicted_masks"
        if not mask_dir.is_dir():
            fail(f"Predicted mask directory is missing: {mask_dir}")
        expected_masks = {Path(name).stem + ".png" for name in test_names}
        if len(expected_masks) != len(test_names):
            fail(f"Original filenames collide after PNG stem conversion in {fold_dir}")
        actual_masks = {path.name for path in mask_dir.iterdir()
                        if path.is_file() and path.suffix.lower() == ".png"}
        if actual_masks != expected_masks:
            missing = sorted(expected_masks - actual_masks)[:8]
            extra = sorted(actual_masks - expected_masks)[:8]
            fail(f"Masks differ from outer_test in {fold_dir}: missing={missing}, extra={extra}")

        for name in test_names:
            all_test_counts[name] = all_test_counts.get(name, 0) + 1
            if name in chosen:
                mask_path = mask_dir / (Path(name).stem + ".png")
                width, height = png_size(mask_path)
                outer_test_matches[name].append(
                    PreparedImage(chosen[name], fold, coords[name], mask_path,
                                  width, height))
    if any(names != split_name_sets[0] for names in split_name_sets[1:]):
        fail("Fold patient_split.csv files do not describe the same image inventory")
    invalid_test = [name for name, count in all_test_counts.items() if count != 1]
    missing_test = split_name_sets[0] - set(all_test_counts)
    if invalid_test or missing_test:
        fail(f"The run is not one-time outer-test OOF: repeated={invalid_test[:8]}, "
             f"never_test={sorted(missing_test)[:8]}")
    invalid_selected = [name for name, matches in outer_test_matches.items()
                        if len(matches) != 1]
    if invalid_selected:
        fail(f"Selected images do not have exactly one outer-test prediction: {invalid_selected[:8]}")
    return [outer_test_matches[image.filename][0] for image in selected], str(manifest.get("run_id", ""))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyzer_sha256(analyzer_dir: Path) -> str:
    digest = hashlib.sha256()
    sources = [(name, analyzer_dir / name)
               for name in ("analyzer.py", "core_algorithm.py", "utils.py")]
    sources.append(("section4_oof_limb_export.py", Path(__file__).resolve()))
    for name, path in sources:
        if not path.is_file():
            fail(f"Analyzer source is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_analyzer(analyzer_dir: Path):
    analyzer_path = analyzer_dir / "analyzer.py"
    # The analyzer lives in a separate, read-only project. Importing it must
    # not create or update __pycache__ files beside the source.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(analyzer_dir))
    spec = importlib.util.spec_from_file_location("section4_limb_analyzer", analyzer_path)
    if spec is None or spec.loader is None:
        fail(f"Could not import analyzer: {analyzer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze_single_image


def finite_coordinate_values(prepared: PreparedImage) -> tuple[float, float, float, float, float]:
    row = prepared.coordinates
    values = tuple(float(row[column]) for column in
                   ("pred_U_x", "pred_U_y", "pred_D_x", "pred_D_y"))
    ux, uy, dx, dy = values
    if not all(map(math.isfinite, values)):
        fail("Non-finite predicted U/D coordinate")
    if not (0 <= ux < prepared.mask_width and 0 <= dx < prepared.mask_width and
            0 <= uy < prepared.mask_height and 0 <= dy < prepared.mask_height):
        fail(f"Predicted U/D coordinates outside {prepared.mask_width}x{prepared.mask_height} mask")
    diameter = math.hypot(dx - ux, dy - uy)
    if diameter <= 0:
        fail("Predicted U and D are coincident")
    saved = str(row.get("pred_length_px") or "").strip()
    if saved:
        saved_diameter = float(saved)
        if not math.isfinite(saved_diameter) or not math.isclose(
                diameter, saved_diameter, rel_tol=1e-4, abs_tol=0.01):
            fail(f"pred_length_px ({saved}) differs from predicted U/D distance ({diameter:.4f})")
    return ux, uy, dx, dy, diameter


def branch_measurement(leg: dict) -> tuple[float, float, float, float, float]:
    diameter = float(leg["max_d"])
    cx, cy = float(leg["max_x"]), float(leg["max_y"])
    px, py = map(float, leg["local_perp_v"])
    pos, neg = float(leg["max_d_pos"]), float(leg["max_d_neg"])
    if not all(math.isfinite(v) for v in (diameter, cx, cy, px, py, pos, neg)):
        fail("Non-finite branch measurement")
    if diameter <= 0 or pos < 0 or neg < 0:
        fail("Non-positive branch diameter or ray distance")
    if not math.isclose(math.hypot(px, py), 1.0, rel_tol=1e-3, abs_tol=1e-3):
        fail("Branch perpendicular vector is not a unit vector")
    if not math.isclose(pos + neg, diameter, rel_tol=1e-3, abs_tol=0.05):
        fail("Branch ray distances do not reproduce max_d")
    x1, y1 = cx + px * pos, cy + py * pos
    x2, y2 = cx - px * neg, cy - py * neg
    if not math.isclose(math.hypot(x1 - x2, y1 - y2), diameter,
                        rel_tol=1e-3, abs_tol=0.05):
        fail("Exported branch line does not reproduce max_d")
    return diameter, x1, y1, x2, y2


def export_rows(prepared: list[PreparedImage], run_id: str,
                analyzer_dir: Path) -> list[dict[str, object]]:
    try:
        import pandas as pd
        analyze_single_image = load_analyzer(analyzer_dir)
    except ImportError as exc:
        raise RuntimeError("Full export requires pandas and the analyzer dependencies "
                           "(numpy, scipy, scikit-image, OpenCV) in this Python environment") from exc

    source_hash = analyzer_sha256(analyzer_dir)
    rows: list[dict[str, object]] = []
    for item in prepared:
        record: dict[str, object] = {name: "" for name in FIELDNAMES}
        record.update({
            "Filename": item.selected.filename,
            "category": item.selected.category,
            "patient_id": item.selected.patient_id,
            "fold": item.fold,
            "Status": "Failed",
            "Run_ID": run_id,
            "Algorithm_SHA256": source_hash,
            "Predicted_Mask": str(item.mask_path),
            "Mask_SHA256": file_sha256(item.mask_path),
        })
        rows.append(record)
        try:
            ux, uy, dx, dy, loop_diameter = finite_coordinate_values(item)
            record.update({"Pred_U_x": ux, "Pred_U_y": uy,
                           "Pred_D_x": dx, "Pred_D_y": dy,
                           "Loop_Diameter_px": loop_diameter})
            keypoints = pd.DataFrame([{"filename": item.selected.filename,
                                       "U_x": ux, "U_y": uy, "D_x": dx, "D_y": dy}])
            result = analyze_single_image(str(item.mask_path), keypoints)
            if not result.get("ok"):
                fail(str(result.get("reason", "Analyzer returned ok=False")))
            legs = sorted(result.get("leg_results", []), key=lambda leg: float(leg["mean_x"]))
            if len(legs) != 2:
                fail(f"Expected two measurable branches, found {len(legs)}")
            measured = []
            quality_warnings = []
            for number, leg in enumerate(legs, start=1):
                width, x1, y1, x2, y2 = branch_measurement(leg)
                side = "Left" if number == 1 else "Right"
                path_points = int(leg["path_points"])
                topology_points = int(leg["topology_eligible_points"])
                ray_valid_points = int(leg["ray_valid_points"])
                edt_fallback = bool(leg["used_edt_fallback"])
                if not (0 <= ray_valid_points <= topology_points <= path_points):
                    fail(f"Invalid support counts for branch {number}")
                record.update({
                    f"Branch{number}_ID": int(leg["id"]),
                    f"Branch{number}_Side": side,
                    f"Branch{number}_Diameter_px": width,
                    f"Branch{number}_x1": x1,
                    f"Branch{number}_y1": y1,
                    f"Branch{number}_x2": x2,
                    f"Branch{number}_y2": y2,
                    f"Branch{number}_Path_Points": path_points,
                    f"Branch{number}_Topology_Candidates": topology_points,
                    f"Branch{number}_Ray_Valid_Points": ray_valid_points,
                    f"Branch{number}_EDT_Fallback": edt_fallback,
                })
                measured.append(width)
                if edt_fallback:
                    quality_warnings.append(f"Branch {number} used EDT fallback")
                elif ray_valid_points < MIN_VALID_RAY_POINTS:
                    quality_warnings.append(
                        f"Branch {number} has only {ray_valid_points} ray-valid points")
            if measured[0] == measured[1]:
                arterial_branch = venous_branch = "Tie"
            else:
                arterial_branch = 1 if measured[0] < measured[1] else 2
                venous_branch = 2 if arterial_branch == 1 else 1
            path_lengths = [int(leg["path_points"]) for leg in legs]
            if min(path_lengths) <= 0:
                fail("Both branch paths must have positive length")
            path_balance = min(path_lengths) / max(path_lengths)
            if path_balance < MIN_BRANCH_PATH_BALANCE:
                quality_warnings.append(
                    f"Branch path imbalance {path_balance:.3f} (<1/3); one limb may be incomplete")
            record.update({
                "Branch_Path_Balance": path_balance,
                "Estimated_Arterial_Diameter_px": min(measured),
                "Estimated_Arterial_Branch": arterial_branch,
                "Estimated_Venous_Diameter_px": max(measured),
                "Estimated_Venous_Branch": venous_branch,
                "Status": "Uncertain" if quality_warnings else "Success",
                "Reason": "; ".join(quality_warnings),
            })
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            record["Reason"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            record["Reason"] = f"AnalyzerError({type(exc).__name__}): {exc}"
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Completed MTL run containing fold_*/ patient split, coordinates, masks")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--selection-dir", type=Path,
                           help=f"Category directories with selected images (default: {DEFAULT_SELECTION_DIR})")
    selection.add_argument("--selection-csv", type=Path,
                           help="CSV with Filename (and optional category) for selected images")
    parser.add_argument("--expected-count", type=int, default=51,
                        help="Required number of selected images; 0 disables the count check (default: 51)")
    parser.add_argument("--analyzer-dir", type=Path, default=DEFAULT_ANALYZER_DIR,
                        help=f"Read-only limb analyzer source (default: {DEFAULT_ANALYZER_DIR})")
    parser.add_argument("--output", type=Path,
                        help="Output CSV inside this workspace's check directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify OOF provenance and mask/coordinate inventory; write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing export CSV at --output")
    args = parser.parse_args(argv)
    if args.expected_count < 0:
        parser.error("--expected-count must be nonnegative")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")
    selected = select_images(args.selection_dir, args.selection_csv, args.expected_count)
    prepared, run_id = verify_run(run_dir, selected)
    fold_counts: dict[int, int] = {}
    for item in prepared:
        fold_counts[item.fold] = fold_counts.get(item.fold, 0) + 1
    print(f"Verified {len(prepared)} selected outer-test images from "
          f"{len(fold_counts)} folds, {len({x.selected.patient_id for x in prepared})} patients")
    print(f"Run: {run_dir} (id={run_id}); selected fold counts: {fold_counts}")
    if args.dry_run:
        bad_coordinates = []
        for item in prepared:
            try:
                finite_coordinate_values(item)
            except (ValueError, TypeError) as exc:
                bad_coordinates.append((item.selected.filename, str(exc)))
        print(f"Coordinate-frame checks: {len(prepared) - len(bad_coordinates)} valid, "
              f"{len(bad_coordinates)} invalid")
        for name, reason in bad_coordinates[:8]:
            print(f"  {name}: {reason}")
        print("Dry run complete; no analysis imported and no file written")
        return 0

    output = args.output or (WORKSPACE / "check" /
                             f"section4_oof_limb_{run_dir.name}.csv")
    output = output.resolve()
    allowed = (WORKSPACE / "check").resolve()
    if not output.is_relative_to(allowed) or output.suffix.lower() != ".csv":
        parser.error(f"Output must be a CSV inside {allowed}")
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists; pass --overwrite to replace it: {output}")
    rows = export_rows(prepared, run_id, args.analyzer_dir.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(row["Status"] == status for row in rows)
              for status in ("Success", "Uncertain", "Failed")}
    print(f"Exported {len(rows)} rows ({counts}): {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
