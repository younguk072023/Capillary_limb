"""Export five thesis examples from a completed DINOv3 run, without inference.

Ground truth: cyan X for both U and D. Prediction: yellow hollow circle.
Reads saved RGB images, heatmaps and coordinates only. No torch/model import.
Outputs live under Capillary_limb/limb/outputs, outside the training workspace.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

DESKTOP = Path.home() / "Desktop"
LIMB = DESKTOP / "Capillary_limb" / "limb"
DEFAULT_RUN = (DESKTOP / "논문 결과" /
               "results_unet_mtl_dinov3_vitb16_finetune_heatmap_dbc65ae36d7e")
EXAMPLES = (
    (5, "p16_det_031_crop_5.png"),
    (5, "p16_det_036_crop_0.png"),
    (5, "p6_det_036.png"),
    (5, "p18_det_070_crop_3.png"),
    (4, "p23_det_007.png"),
)
GT_COLOR = "#00E5FF"
PRED_COLOR = "#FFFF00"
UM_PER_PIXEL = 0.55


class Sources:
    def __init__(self):
        self.hashes = {}

    def read(self, path: Path) -> bytes:
        path = path.resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if path in self.hashes and self.hashes[path] != digest:
            raise RuntimeError(f"Input changed during export: {path}")
        self.hashes[path] = digest
        return data

    def json(self, path):
        return json.loads(self.read(path).decode("utf-8-sig"))

    def table(self, path):
        rows = list(csv.DictReader(io.StringIO(self.read(path).decode("utf-8-sig"))))
        indexed = {row["filename"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate filenames: {path}")
        return indexed

    def image(self, path):
        with Image.open(io.BytesIO(self.read(path))) as image:
            return image.convert("RGB")

    def verify_unchanged(self):
        for path, digest in self.hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Input changed during export: {path}")


def xy(row, prefix):
    return np.array([[float(row[f"{prefix}_{point}_{axis}"])
                      for axis in ("x", "y")] for point in ("U", "D")])


def load_examples(run: Path, sources: Sources):
    manifest = sources.json(run / "experiment_manifest.json")
    spec = manifest["run_spec"]
    model = spec["experiment"]
    if (spec["n_splits"] != 5 or model["model_name"] != "unet" or
            model["encoder_name"] != "dinov3" or model["task_mode"] != "mtl" or
            model["kpt_mode"] != "heatmap"):
        raise ValueError("Expected a completed five-fold U-Net MTL DINOv3 heatmap run")
    master = sources.table(run / "master_predicted_coordinates.csv")
    splits = {fold: sources.table(run / f"fold_{fold}" / "patient_split.csv")
              for fold in range(1, 6)}
    fold_rows = {fold: sources.table(run / f"fold_{fold}" / "predicted_coordinates.csv")
                 for fold in (4, 5)}
    evaluated = {fold: sources.table(run / f"fold_{fold}" / "evaluated_images.csv")
                 for fold in (4, 5)}
    examples = []
    for fold, name in EXAMPLES:
        row = master[name]
        if int(row["fold"]) != fold or name not in evaluated[fold]:
            raise ValueError(f"Unexpected evaluation fold: {name}")
        test_folds = [f for f in splits if splits[f][name]["split"] == "outer_test"]
        if test_folds != [fold]:
            raise ValueError(f"Expected exactly one matching outer-test fold: {name}")
        patient = name.split("_det", 1)[0]
        if any(r["split"] != "outer_test" for r in splits[fold].values()
               if r["patient_id"] == patient):
            raise ValueError(f"Patient is not entirely held out: {name}")
        gt, pred = xy(row, "gt"), xy(row, "pred")
        for prefix, values in (("gt", gt), ("pred", pred)):
            if not np.allclose(values, xy(fold_rows[fold][name], prefix), atol=1e-6, rtol=0):
                raise ValueError(f"Master/fold coordinate mismatch: {name}")
            if not np.isfinite(values).all():
                raise ValueError(f"Missing coordinates: {name}")
        directory = run / f"fold_{fold}" / "heatmap_outputs" / Path(name).stem
        original = sources.image(directory / "original.png")
        meta = sources.json(directory / "visualization.json")
        caller = meta["caller_metadata"]
        if caller["run_id"] != manifest["run_id"] or caller["fold"] != fold or caller["filename"] != name:
            raise ValueError(f"Saved visualization belongs to a different run: {name}")
        width, height = original.size
        if [height, width] != list(spec["resize_shape"]):
            raise ValueError(f"Image and coordinate dimensions disagree: {name}")
        with np.load(io.BytesIO(sources.read(directory / "maps.npz")), allow_pickle=False) as data:
            for key, points in (("reference_xy", gt), ("prediction_xy", pred)):
                if not np.allclose(data[key], points, atol=1e-6, rtol=0):
                    raise ValueError(f"Saved heatmap/coordinate mismatch: {name}/{key}")
        for points in (gt, pred):
            if (points < 0).any() or (points >= [width, height]).any():
                raise ValueError(f"Out-of-frame coordinates: {name}")
        heatmaps = [sources.image(run / f"fold_{fold}" / "predicted_heatmaps" /
                                  f"{Path(name).stem}_heatmap_{channel}.png")
                    for channel in ("U", "D")]
        if any(image.size != original.size for image in heatmaps):
            raise ValueError(f"Heatmap dimensions disagree: {name}")
        examples.append({"fold": fold, "filename": name, "gt": gt, "pred": pred,
                         "original": original, "heatmaps": heatmaps})
    if len({example["original"].size for example in examples}) != 1:
        raise ValueError("The five examples must have matching frame sizes")
    return manifest, examples


def mark(image, gt=None, pred=None, *, gt_color=GT_COLOR, pred_color=PRED_COLOR,
         radius=6.0):
    """Antialias marker layers only; retain original RGB outside the markers."""
    scale = 4
    layer = Image.new("RGBA", (image.width * scale, image.height * scale))
    draw = ImageDraw.Draw(layer)
    # Draw hollow predictions first, then GT crosses so both remain visible.
    for points, kind, color in ((pred, "circle", pred_color), (gt, "x", gt_color)):
        if points is None:
            continue
        for x, y in points:
            x, y, r = float(x) * scale, float(y) * scale, radius * scale
            if kind == "circle":
                box = (x-r, y-r, x+r, y+r)
                draw.ellipse(box, outline=(20, 20, 20, 220), width=12)
                draw.ellipse(box, outline=color, width=8)
            else:
                for line in ((x-r, y-r, x+r, y+r), (x-r, y+r, x+r, y-r)):
                    draw.line(line, fill=(20, 20, 20, 220), width=14)
                    draw.line(line, fill=color, width=9)
    layer = layer.resize(image.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def font(size, bold=False):
    path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("timesbd.ttf" if bold else "times.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def montage(rows, labels, *, gt_color, pred_color):
    """Assemble native-resolution panels without altering spatial coordinates."""
    tile_w, tile_h = rows[0][0].size
    gap, left, top, right, bottom = 16, 330, 64, 20, 104
    canvas = Image.new("RGB", (left + 5*tile_w + 4*gap + right,
                               top + len(rows)*tile_h + (len(rows)-1)*gap + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    for column in range(5):
        draw.text((left + column*(tile_w+gap) + tile_w/2, top/2),
                  f"({chr(97+column)})", font=font(42, True), fill="black", anchor="mm")
    for index, (images, label) in enumerate(zip(rows, labels)):
        y = top + index*(tile_h+gap)
        draw.multiline_text((left/2, y+tile_h/2), label, font=font(40, True),
                            fill="black", anchor="mm", align="center", spacing=10)
        for column, image in enumerate(images):
            canvas.paste(image, (left+column*(tile_w+gap), y))
    legend_y = canvas.height - 52
    # Symbols use the same colors for U and D in every panel.
    legend = Image.new("RGB", (80, 70), "white")
    legend = mark(legend, gt=[[40, 35]], gt_color=gt_color, radius=15)
    canvas.paste(legend, (left+550, legend_y-35))
    draw.text((left+640, legend_y), "Ground truth (X)", font=font(40), fill="black", anchor="lm")
    legend = Image.new("RGB", (80, 70), "white")
    legend = mark(legend, pred=[[40, 35]], pred_color=pred_color, radius=15)
    canvas.paste(legend, (left+1150, legend_y-35))
    draw.text((left+1240, legend_y), "Prediction (circle)", font=font(40), fill="black", anchor="lm")
    return canvas


def save_image(image, path, tiff=False):
    image.save(path.with_suffix(".png"), dpi=(300, 300))
    if tiff:
        image.save(path.with_suffix(".tiff"), dpi=(300, 300), compression="tiff_lzw")


def render(examples, output, args):
    panel_rows = {name: [] for name in ("original", "upper", "lower", "gt", "pred", "comparison", "zoom")}
    records = []
    for index, example in enumerate(examples):
        gt, pred, original = example["gt"], example["pred"], example["original"]
        options = {"gt_color": args.gt_color, "pred_color": args.pred_color, "radius": args.marker_radius}
        upper, lower = example["heatmaps"]
        panels = {"original": original,
                  "upper": mark(upper, gt=gt[:1], **options),
                  "lower": mark(lower, gt=gt[1:], **options),
                  "gt": mark(original, gt=gt, **options),
                  "pred": mark(original, pred=pred, **options),
                  "comparison": mark(original, gt=gt, pred=pred, **options)}
        # All zooms use the same 128x128-pixel field and 4x display magnification.
        points = np.vstack((gt, pred))
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        side = 128
        if np.max(np.ptp(points, axis=0)) + 2*args.marker_radius >= side:
            raise ValueError("Keypoints do not fit the fixed 128-pixel detail panel")
        x0 = int(np.clip(round(center[0]-side/2), 0, original.width-side))
        y0 = int(np.clip(round(center[1]-side/2), 0, original.height-side))
        crop_box = (x0, y0, x0+side, y0+side)
        panels["zoom"] = panels["comparison"].crop(crop_box).resize(original.size, Image.Resampling.LANCZOS)
        directory = output / f"fold_{example['fold']}" / Path(example["filename"]).stem
        directory.mkdir(parents=True)
        filenames = {"original": "original", "upper": "upper_heatmap_gt_x",
                     "lower": "lower_heatmap_gt_x", "gt": "ground_truth_x",
                     "pred": "prediction_circle", "comparison": "gt_prediction_overlay",
                     "zoom": "gt_prediction_overlay_zoom4x"}
        for name, image in panels.items():
            save_image(image, directory / filenames[name])
            panel_rows[name].append(image)
        save_image(upper, directory / "upper_heatmap_saved")
        save_image(lower, directory / "lower_heatmap_saved")
        record = {"panel": chr(97+index), "fold": example["fold"], "filename": example["filename"],
                  "coordinate_unit": "px", "coordinate_order": "U, D; each point is x, y",
                  "image_size": list(original.size), "gt_xy": gt.tolist(), "pred_xy": pred.tolist(),
                  "keypoint_error_px": np.linalg.norm(pred-gt, axis=1).tolist(),
                  "keypoint_error_um": (np.linalg.norm(pred-gt, axis=1)*UM_PER_PIXEL).tolist(),
                  "detail_crop_box_xyxy": list(crop_box), "detail_magnification": original.width/side}
        records.append(record)
        (directory / "coordinates.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    color_options = {"gt_color": args.gt_color, "pred_color": args.pred_color}
    figure = montage([panel_rows[name] for name in ("original", "upper", "lower", "gt", "pred", "comparison")],
                     ["Original image", "Upper heatmap", "Lower heatmap", "Ground truth", "Prediction", "GT + prediction"], **color_options)
    save_image(figure, output / "dinov3_examples_full", tiff=True)
    figure = montage([panel_rows[name] for name in ("gt", "pred", "comparison")],
                     ["Ground truth", "Prediction", "GT + prediction"], **color_options)
    save_image(figure, output / "dinov3_gt_prediction_comparison", tiff=True)
    figure = montage([panel_rows["zoom"]], ["Keypoint detail\n(4x)"], **color_options)
    save_image(figure, output / "dinov3_keypoint_detail_4x", tiff=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, help="Fresh output folder; default: limb/outputs/dinov3_keypoint_examples_5_DATE")
    parser.add_argument("--gt-color", default=GT_COLOR, help="One color for all GT X markers, U and D")
    parser.add_argument("--pred-color", default=PRED_COLOR, help="One color for all predicted circles, U and D")
    parser.add_argument("--marker-radius", type=float, default=6.0, help="Marker radius in original-image pixels")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for color in (args.gt_color, args.pred_color):
        ImageColor.getrgb(color)
    if not np.isfinite(args.marker_radius) or not 1 <= args.marker_radius <= 12:
        parser.error("--marker-radius must be between 1 and 12 pixels")
    run = args.run_dir.resolve()
    sources = Sources()
    manifest, examples = load_examples(run, sources)
    for example in examples:
        print(f"Verified outer-test fold {example['fold']}: {example['filename']}", flush=True)
    if args.dry_run:
        print("Read-only validation passed. No output written.")
        return
    default_output = LIMB / "outputs" / f"dinov3_keypoint_examples_5_{datetime.now():%Y%m%d}"
    output = args.output_dir.resolve() if args.output_dir else default_output
    counter = 2
    while output.exists() and args.output_dir is None:
        output = default_output.with_name(f"{default_output.name}_run{counter}")
        counter += 1
    for protected in (run, DESKTOP / "Multi-Task-Learning", Path(manifest["run_spec"]["data_root"])):
        if output == protected.resolve() or protected.resolve() in output.parents:
            raise ValueError("Output must be outside the model run, training workspace and dataset")
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".keypoint_figures_", dir=output.parent) as temporary:
        stage = Path(temporary)
        records = render(examples, stage, args)
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_dir": str(run), "run_id": manifest["run_id"],
                    "script": str(Path(__file__).resolve()),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "ground_truth_marker": {"shape": "X", "color": args.gt_color, "both_U_and_D": True},
                    "prediction_marker": {"shape": "hollow circle", "color": args.pred_color, "both_U_and_D": True},
                    "marker_radius_px": args.marker_radius,
                    "marker_line_width_px": {"ground_truth": 2.25, "prediction": 2.0},
                    "marker_style": "Original cyan X / yellow hollow circle, original size and layout; colored strokes increased by 0.5 px",
                    "heatmap_display": "Existing JET U/D PNGs reused without recomputation; only the corresponding GT X is added",
                    "background": "Full saved original RGB; no crop or enhancement except explicitly named 4x detail exports",
                    "image_export": "Native panel resolution; 300-dpi PNG and lossless TIFF for assembled figures",
                    "um_per_pixel": UM_PER_PIXEL, "examples": records,
                    "sources": [{"path": str(p), "sha256": h} for p, h in sources.hashes.items()]}
        (stage / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        mapping = "\n".join(f"({r['panel']}) Fold {r['fold']}: {r['filename']}" for r in records)
        readme = f"""DINOv3 논문용 좌표 비교 그림

1. dinov3_examples_full.png / .tiff
   행 순서: 원본, Upper heatmap, Lower heatmap, GT, 예측, GT+예측.
2. dinov3_gt_prediction_comparison.png / .tiff
   행 순서: GT, 예측, GT+예측.
3. dinov3_keypoint_detail_4x.png / .tiff
   같은 128×128 px 영역 크기를 4배 확대한 정점 비교. 별도 확대 그림입니다.

모든 GT U/D: 같은 청록색 X ({args.gt_color}).
모든 예측 U/D: 같은 노란색 빈 원 ({args.pred_color}).
처음 색상·크기·배치를 유지하고 선 두께만 0.5 px 늘렸습니다 (GT 2.25 px, 예측 2 px).
Upper heatmap에는 GT U만, Lower heatmap에는 GT D만 X로 표시합니다.
히트맵은 해당 Fold에 저장된 JET PNG를 그대로 사용합니다.
전체 사진은 자르거나 대비를 조절하지 않았습니다.
각 사진의 독립 PNG는 fold_4 또는 fold_5 하위 폴더에 있습니다.
숫자 좌표와 실제 입력 SHA256은 coordinates.json / provenance.json에 있습니다.
MTL 학습 코드, 모델 파일, 원본 데이터, 저장된 결과는 읽기만 합니다.
학습/추론을 실행하거나 GPU를 사용하지 않습니다.

왼쪽부터:
{mapping}
"""
        (stage / "README.txt").write_text(readme, encoding="utf-8-sig")
        sources.verify_unchanged()
        shutil.copytree(stage, output)
    print(f"Saved: {output}", flush=True)
    print(f"GT = X ({args.gt_color}); prediction = hollow circle ({args.pred_color}). All five examples verified.", flush=True)


if __name__ == "__main__":
    main()
