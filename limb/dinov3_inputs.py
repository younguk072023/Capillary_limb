"""Validated DINOv3 inputs for the 141-image viewer and geometry CSV exporter.

Only predicted U/D coordinates are adapted to the analyzer's historical column
names. This module deliberately does not import plotting or statistical code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


DESKTOP = Path.home() / "Desktop"
DEFAULT_RUN = (DESKTOP / "논문 결과" /
               "results_unet_mtl_dinov3_vitb16_finetune_heatmap_dbc65ae36d7e")
DEFAULT_SELECTION = DESKTOP / "Capillary_limb" / "check" / "diameter_selection_141.csv"
DEFAULT_PREDICTED = DESKTOP / "dinov3_segmentation"
DEFAULT_GT = DESKTOP / "GT_masks_144"
OUTPUT_ROOT = DESKTOP / "Multi-Task-Learning" / "outputs"
PRED_FOLDERS = {"U": "U", "tortuous": "tortuous", "twist": "crossing"}
COORDINATES = ("pred_U_x", "pred_U_y", "pred_D_x", "pred_D_y")


@dataclass
class DINOInputs:
    root: Path
    image_files: list[str]
    keypoints: pd.DataFrame
    coordinate_csv: Path
    selection_csv: Path
    run_dir: Path
    run_id: str
    mask_source: str


def add_input_arguments(parser) -> None:
    parser.add_argument("--mask-source", choices=("predicted", "gt"), default="predicted")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="Override the selected source's mask root")
    parser.add_argument("--check", action="store_true",
                        help="Validate all 141 inputs without opening a viewer or writing results")


def _read_unique(path: Path, columns: tuple[str, ...], key: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={key: str})
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(set(columns) - set(frame.columns))}")
    if frame[key].isna().any() or not frame[key].is_unique:
        raise ValueError(f"Missing or duplicate {key} in {path}")
    return frame.set_index(key, drop=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(args) -> DINOInputs:
    run_dir = args.run_dir.resolve()
    selection_csv = args.selection_csv.resolve()
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8-sig"))
    spec = manifest["run_spec"]
    experiment = spec["experiment"]
    if (experiment.get("model_name") != "unet" or experiment.get("encoder_name") != "dinov3"
            or experiment.get("task_mode") != "mtl" or spec.get("n_splits") != 5):
        raise ValueError("Expected the completed five-fold U-Net MTL DINOv3 run")
    if not manifest.get("run_id"):
        raise ValueError("The run manifest has no run ID")
    height, width = map(int, spec["resize_shape"])
    selection = _read_unique(selection_csv, ("Filename", "Category"), "Filename")
    if len(selection) != 141 or Counter(selection["Category"]) != Counter({name: 47 for name in PRED_FOLDERS}):
        raise ValueError("Expected 141 selected images: U 47, tortuous 47, twist/Crossing 47")
    names = selection["Filename"].tolist()
    if any(Path(name).name != name or "/" in name or "\\" in name for name in names):
        raise ValueError("Selection CSV must contain bare image filenames")
    if len({Path(name).stem.casefold() for name in names}) != 141:
        raise ValueError("Selected filename stems must be unique")
    coordinate_csv = run_dir / "master_predicted_coordinates.csv"
    master = _read_unique(coordinate_csv, ("filename", "fold", *COORDINATES, "pred_length_px"), "filename")
    if not set(names).issubset(master.index):
        raise ValueError("Selected images are missing from the DINOv3 master coordinate CSV")
    folds = {}
    tests = Counter()
    for fold in range(1, 6):
        directory = run_dir / f"fold_{fold}"
        frame = _read_unique(directory / "predicted_coordinates.csv",
                             ("filename", *COORDINATES), "filename")
        split = _read_unique(directory / "patient_split.csv", ("filename", "split"), "filename")
        if not set(names).issubset(split.index):
            raise ValueError(f"Selected images missing from fold {fold} patient split")
        for name in names:
            if str(split.loc[name, "split"]).strip().lower() == "outer_test":
                tests[name] += 1
                if name not in frame.index or float(master.loc[name, "fold"]) != fold:
                    raise ValueError(f"Master/outer-test fold mismatch: {name}")
        folds[fold] = frame
    if any(tests[name] != 1 for name in names):
        raise ValueError("Each selected image must have exactly one outer-test fold")
    root = (args.mask_dir or (DEFAULT_PREDICTED if args.mask_source == "predicted" else DEFAULT_GT)).resolve()
    gt_manifest = None
    if args.mask_source == "gt":
        gt_manifest = _read_unique(root / "gt_mask_manifest_141.csv",
                                   ("Selected_Image", "Copy_SHA256", "Category"), "Selected_Image")
        gt_manifest = gt_manifest.assign(Filename=gt_manifest["Selected_Image"].map(lambda value: Path(value).name))
        if set(gt_manifest["Filename"]) != set(names) or not gt_manifest["Filename"].is_unique:
            raise ValueError("GT manifest does not match the selected 141 images")
        gt_manifest = gt_manifest.set_index("Filename")
    image_files = []
    adapted = []
    for row in selection.itertuples(index=False):
        name, category = row.Filename, row.Category
        coordinate = master.loc[name]
        fold = int(coordinate["fold"])
        values = [float(coordinate[column]) for column in COORDINATES]
        if not all(math.isfinite(value) and 0 <= value < bound
                   for value, bound in zip(values, (width, height, width, height))):
            raise ValueError(f"Invalid predicted coordinates for {name}")
        for column, value in zip(COORDINATES, values):
            if not math.isclose(value, float(folds[fold].loc[name, column]), abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"Master/fold predicted coordinate mismatch: {name}, {column}")
        diameter = math.hypot(values[2] - values[0], values[3] - values[1])
        if diameter <= 0 or not math.isclose(diameter, float(coordinate["pred_length_px"]), abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"Predicted U/D length mismatch: {name}")
        if args.mask_source == "predicted":
            relative = Path(PRED_FOLDERS[category]) / (Path(name).stem + ".png")
            original = run_dir / f"fold_{fold}" / "predicted_masks" / (Path(name).stem + ".png")
            if _sha256(root / relative) != _sha256(original):
                raise ValueError(f"Predicted mask differs from its outer-test fold: {name}")
        else:
            relative = Path(category) / (Path(name).stem + ".tif")
            if str(gt_manifest.loc[name, "Category"]) != category:
                raise ValueError(f"GT category mismatch: {name}")
            if _sha256(root / relative).lower() != str(gt_manifest.loc[name, "Copy_SHA256"]).lower():
                raise ValueError(f"GT mask differs from its manifest: {name}")
        image_files.append(str(relative))
        adapted.append(dict(zip(("filename", "U_x", "U_y", "D_x", "D_y"), (name, *values))))
    return DINOInputs(root, image_files, pd.DataFrame(adapted), coordinate_csv,
                      selection_csv, run_dir, str(manifest["run_id"]), args.mask_source)


def print_inputs(inputs: DINOInputs) -> None:
    print(f"DINOv3 run: {inputs.run_id}")
    print(f"Mask source: {inputs.mask_source}; root: {inputs.root}")
    print(f"Selection: {inputs.selection_csv}")
    print(f"Validated {len(inputs.image_files)} images (47 U, 47 Tortuous, 47 Crossing)")
    print(f"Predicted U/D: {inputs.coordinate_csv}")
