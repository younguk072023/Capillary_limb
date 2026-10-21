"""DINOv3 141-image diameter export and mask overlays.

The default uses DINOv3 predicted masks and predicted U/D coordinates from each
image's outer-test fold. Use --mask-source gt for GT masks with the same U/D.
All finite positive branch pairs are exported: A is the thinner branch and V
the wider branch (estimated sides, not anatomical labels). Status describes
measurement availability; QC_Status preserves the analyzer's quality status.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import textwrap

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
LIMB_DIR = Path.home() / "Desktop" / "Capillary_limb" / "limb"
if not (SCRIPT_DIR / "dinov3_inputs.py").is_file():
    sys.path.insert(0, str(LIMB_DIR))
from dinov3_inputs import add_input_arguments, load_inputs, print_inputs

CHECK_DIR = LIMB_DIR.parent / "check"
sys.path.insert(0, str(CHECK_DIR))
import section4_oof_limb_export as oof

LEGACY_FIELDS = [
    "Filename", "A_x1", "A_y1", "A_x2", "A_y2",
    "Algo_Arterial_Diameter(px)", "V_x1", "V_y1", "V_x2", "V_y2",
    "Algo_Venous_Diameter(px)", "Status", "QC_Status", "Reason", "Shape",
    "U_x", "U_y", "D_x", "D_y", "Algo_Loop_Diameter(px)",
    "A_Branch", "V_Branch", "Mask_Source", "Mask_Path", "Mask_SHA256",
    "fold", "Run_ID", "Algorithm_SHA256",
]
SHAPE_NAMES = {"u": "u", "tortuous": "tortuous", "twist": "crossing"}


def positive(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (ValueError, TypeError):
        return False


def measurement_row(record: dict, source: str) -> dict:
    """Adapt canonical geometry without rounding the measurement endpoints."""
    row = dict.fromkeys(LEGACY_FIELDS, "")
    row.update({
        "Filename": record["Filename"], "Status": "Failed",
        "QC_Status": record["Status"], "Reason": record["Reason"],
        "Shape": SHAPE_NAMES[str(record["category"]).lower()],
        "U_x": record["Pred_U_x"], "U_y": record["Pred_U_y"],
        "D_x": record["Pred_D_x"], "D_y": record["Pred_D_y"],
        "Algo_Loop_Diameter(px)": record["Loop_Diameter_px"],
        "Mask_Source": source, "Mask_Path": record["Predicted_Mask"],
        "Mask_SHA256": record["Mask_SHA256"], "fold": record["fold"],
        "Run_ID": record["Run_ID"], "Algorithm_SHA256": record["Algorithm_SHA256"],
    })
    if all(positive(record[f"Branch{i}_Diameter_px"]) for i in (1, 2)):
        # Stable branch-number order resolves equal-width ties reproducibly.
        thin, wide = sorted((1, 2), key=lambda i: float(record[f"Branch{i}_Diameter_px"]))
        for label, branch, column in (("A", thin, "Algo_Arterial_Diameter(px)"),
                                      ("V", wide, "Algo_Venous_Diameter(px)")):
            row[column] = record[f"Branch{branch}_Diameter_px"]
            row[f"{label}_Branch"] = branch
            for coordinate in ("x1", "y1", "x2", "y2"):
                row[f"{label}_{coordinate}"] = record[f"Branch{branch}_{coordinate}"]
        row["Status"] = "Success"
    return row


def prepare(inputs):
    selected = oof.select_images(None, inputs.selection_csv, 141)
    prepared, run_id = oof.verify_run(inputs.run_dir, selected)
    if run_id != inputs.run_id:
        raise ValueError("Validated DINOv3 run IDs differ")
    masks = {Path(relative).stem: inputs.root / relative for relative in inputs.image_files}
    prepared = [replace(item, mask_path=masks[Path(item.selected.filename).stem])
                for item in prepared]
    from PIL import Image
    for item in prepared:
        with Image.open(item.mask_path) as mask:
            if mask.size != (item.mask_width, item.mask_height):
                raise ValueError(f"Mask and predicted U/D coordinate frame differ: {item.mask_path}")
    return prepared


def next_output(inputs, requested: Path | None) -> Path:
    if requested is not None:
        output = requested.resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists; choose a new directory: {output}")
        return output
    base = inputs.root / "algorithm_batch_141"
    output = base
    number = 2
    while output.exists():
        output = base.with_name(f"{base.name}_run{number}")
        number += 1
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def font(size: int):
    from PIL import ImageFont
    for candidate in (Path("C:/Windows/Fonts/malgun.ttf"),
                      Path("C:/Windows/Fonts/arial.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_overlay(row: dict, output: Path):
    """Render measured lines and exact-coordinate readouts beside the raw mask."""
    import numpy as np
    from PIL import Image, ImageDraw
    with Image.open(row["Mask_Path"]) as image:
        binary = np.asarray(image.convert("L")) > 127
    height, width = binary.shape
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = (24, 31, 41)
    pixels[binary] = (192, 202, 216)
    panel = Image.new("RGB", (width + 570, max(height + 70, 690)), (24, 31, 41))
    panel.paste(Image.fromarray(pixels), (0, 70))
    draw = ImageDraw.Draw(panel)
    regular, small, title = font(18), font(15), font(21)
    draw.text((16, 10), f"{row['Filename']} | {row['Shape']}", font=title, fill="white")
    draw.text((16, 39), f"DINOv3 | {row['Mask_Source']} mask | fold {row['fold']}",
              font=small, fill=(195, 203, 218))
    x = width + 18
    y = 80
    definitions = [
        ("Loop", "Algo_Loop_Diameter(px)", ("U_x", "U_y", "D_x", "D_y"), (255, 210, 65)),
        ("A / thin branch", "Algo_Arterial_Diameter(px)", ("A_x1", "A_y1", "A_x2", "A_y2"), (50, 218, 240)),
        ("V / wide branch", "Algo_Venous_Diameter(px)", ("V_x1", "V_y1", "V_x2", "V_y2"), (255, 111, 182)),
    ]
    for label, length_column, coordinates, color in definitions:
        available = positive(row[length_column])
        length = f"{float(row[length_column]):.2f} px" if available else "Not available"
        draw.text((x, y), f"{label}: {length}", font=regular, fill=color)
        if available:
            x1, y1, x2, y2 = (float(row[key]) for key in coordinates)
            first = (int(round(x1)), int(round(y1)) + 70)
            second = (int(round(x2)), int(round(y2)) + 70)
            draw.line((first, second), fill=color, width=3)
            for px, py in (first, second):
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=(8, 13, 21), width=1)
            short_label = label.split(" /")[0]
            draw.text((first[0] + 6, first[1] - 20), f"{short_label} {length}", font=small, fill=color,
                      stroke_width=2, stroke_fill=(12, 18, 28))
            draw.text((x, y + 32), f"P1 = ({x1:.3f}, {y1:.3f})", font=small, fill="white")
            draw.text((x, y + 58), f"P2 = ({x2:.3f}, {y2:.3f})", font=small, fill="white")
        y += 112
    draw.text((x, y), f"Measured pair: {row['Status']}", font=regular, fill="white")
    draw.text((x, y + 30), f"Analyzer QC: {row['QC_Status']}", font=regular, fill="white")
    reason = str(row["Reason"]).strip()
    if reason:
        draw.multiline_text((x, y + 64), "\n".join(textwrap.wrap(reason, 55)), font=small,
                            fill=(235, 181, 116), spacing=5)
    draw.text((x, panel.height - 43), "A/V: width-ranked estimated sides", font=small,
              fill=(195, 203, 218))
    panel.save(output)
    return panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    inputs = load_inputs(args)
    print_inputs(inputs)
    prepared = prepare(inputs)
    print(f"Validated all {len(prepared)} mask frames and outer-test assignments", flush=True)
    if args.check:
        print("Check complete. No results written.")
        return 0
    output = next_output(inputs, args.output_dir)
    analyzer_dir = SCRIPT_DIR if (SCRIPT_DIR / "analyzer.py").is_file() else LIMB_DIR
    before_algorithm = oof.analyzer_sha256(analyzer_dir)
    input_hashes = {str(item.mask_path): oof.file_sha256(item.mask_path) for item in prepared}
    print(f"Measuring {len(prepared)} masks with the current limb algorithm...", flush=True)
    canonical = oof.export_rows(prepared, inputs.run_id, analyzer_dir)
    if oof.analyzer_sha256(analyzer_dir) != before_algorithm:
        raise RuntimeError("Algorithm changed during measurement; no results saved")
    for record in canonical:
        if (record["Mask_SHA256"] != input_hashes[record["Predicted_Mask"]] or
                oof.file_sha256(Path(record["Predicted_Mask"])) != record["Mask_SHA256"]):
            raise RuntimeError("An input mask changed during measurement; no results saved")
    rows = [measurement_row(record, inputs.mask_source) for record in canonical]
    if len(rows) != 141 or len({row["Filename"] for row in rows}) != 141:
        raise RuntimeError("Measurement export does not preserve the selected 141 images")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "algorithm_measurements_141.csv", rows, LEGACY_FIELDS)
    write_csv(output / "canonical_measurements_141.csv", canonical, oof.FIELDNAMES)
    for shape in ("u", "tortuous", "crossing"):
        group = [row for row in rows if row["Shape"] == shape]
        write_csv(output / f"algorithm_measurements_141_{shape}.csv", group, LEGACY_FIELDS)
        (output / "visuals" / shape).mkdir(parents=True)
    failed_dir = output / "failed_samples"
    failed_dir.mkdir()
    for row in rows:
        name = f"{Path(row['Filename']).stem}_auto.png"
        panel = draw_overlay(row, output / "visuals" / row["Shape"] / name)
        if row["Status"] == "Failed":
            panel.save(failed_dir / name)
    counts = dict(Counter(row["Status"] for row in rows))
    qc_counts = dict(Counter(row["QC_Status"] for row in rows))
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoder_name": "dinov3", "run_id": inputs.run_id, "run_dir": str(inputs.run_dir),
        "mask_source": inputs.mask_source, "mask_root": str(inputs.root),
        "selection_csv": str(inputs.selection_csv),
        "selection_sha256": oof.file_sha256(inputs.selection_csv),
        "coordinate_csv": str(inputs.coordinate_csv),
        "coordinate_sha256": oof.file_sha256(inputs.coordinate_csv),
        "selected_n": len(rows), "shape_counts": dict(Counter(row["Shape"] for row in rows)),
        "measurement_status_counts": counts, "analyzer_qc_counts": qc_counts,
        "analyzer_dir": str(analyzer_dir), "algorithm_sha256": before_algorithm,
        "status_definition": "Success means both branch diameters are finite and positive; QC_Status preserves analyzer quality status",
        "branch_definition": "A = thin branch; V = wide branch; estimated sides rather than anatomical identities",
        "loop_definition": "Euclidean distance between DINOv3 outer-test predicted U/D",
        "coordinate_definition": "CSV endpoints and diameters retain floating precision; only visual display is rounded",
        "canonical_note": "Canonical Predicted_Mask is the analyzed mask path; for --mask-source gt this points to GT TIFFs",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} records and overlays: {output}")
    print(f"Measurement status: {counts}; analyzer QC: {qc_counts}")
    print(f"CSV: {output / 'algorithm_measurements_141.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
