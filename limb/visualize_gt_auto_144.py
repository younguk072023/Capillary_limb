"""Draw saved automatic diameter lines on the selected DINOv3-evaluated GT masks.

Each PNG shows the GT segmentation, measured U/D loop line, and the two
algorithm-selected branch width lines. Endpoint dots, native 512x512 pixel
coordinates, and lengths are shown. No manual measurements are involved.

The lines are read from the GT evaluation export, not estimated from a new
analyzer run. This keeps every overlay aligned with the saved evaluation even
if the analyzer parameters have subsequently changed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps


DESKTOP = Path.home() / "Desktop"
WORKSPACE = DESKTOP / "Multi-Task-Learning"
DEFAULT_RESULTS = (WORKSPACE / "outputs" /
                   "gt_measurements_dinov3_141_20261007")
DEFAULT_OUTPUT = DESKTOP / "GT_masks_144" / "algorithm_measurements_141"
FRAME = 512
SCALE = 2
MASK_LEFT, MASK_TOP = 35, 105
MASK_DISPLAY = FRAME * SCALE
INFO_LEFT = MASK_LEFT + MASK_DISPLAY + 37
CANVAS_WIDTH, CANVAS_HEIGHT = 1570, 1190
BG = (22, 27, 37)
WHITE = (239, 243, 249)
GRAY = (169, 180, 196)
COLORS = {
    "loop": (255, 198, 61),
    "thin": (50, 217, 240),
    "wide": (255, 105, 157),
}


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_name = {}
    for row in rows:
        name = row["Filename"].strip()
        if not name or Path(name).name != name or name in by_name:
            raise ValueError(f"Invalid or duplicate filename: {name!r}")
        by_name[name] = row
    return by_name


def number(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column} for {row['Filename']}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_line(row: dict[str, str], label: str) -> dict:
    if label == "loop":
        endpoints = (number(row, "Pred_U_x"), number(row, "Pred_U_y"),
                     number(row, "Pred_D_x"), number(row, "Pred_D_y"))
        diameter = number(row, "Loop_Diameter_px")
        fallback = False
    else:
        endpoints = tuple(number(row, f"{label}_{column}")
                          for column in ("x1", "y1", "x2", "y2"))
        diameter = number(row, f"{label}_Diameter_px")
        fallback = row[f"{label}_EDT_Fallback"].strip().lower() == "true"
    x1, y1, x2, y2 = endpoints
    if not all(0 <= coordinate < FRAME for coordinate in endpoints):
        raise ValueError(f"Endpoint outside GT frame: {row['Filename']} {label}")
    if diameter <= 0 or not math.isclose(math.hypot(x2 - x1, y2 - y1),
                                          diameter, rel_tol=1e-4, abs_tol=0.05):
        raise ValueError(f"Saved length does not match endpoints: "
                         f"{row['Filename']} {label}")
    return {"points": endpoints, "length": diameter, "fallback": fallback}


def parse_record(row: dict[str, str]) -> dict:
    name = row["Filename"]
    mask = Path(row["GT_Mask"])
    if not mask.is_file() or sha256(mask) != row["GT_Mask_SHA256"].strip().lower():
        raise ValueError(f"GT mask missing or changed since evaluation: {mask}")
    with Image.open(mask) as source:
        if source.size != (FRAME, FRAME) or getattr(source, "n_frames", 1) != 1:
            raise ValueError(f"Unexpected GT mask dimensions/frames: {mask}")
    if row["Status"] not in {"Success", "Uncertain"}:
        raise ValueError(f"No complete automatic measurement: {name}")
    branches = [parse_line(row, "Branch1"), parse_line(row, "Branch2")]
    branches.sort(key=lambda branch: branch["length"])
    category = row["category"].lower()
    shape = "crossing" if category == "twist" else category
    if shape not in {"u", "tortuous", "crossing"}:
        raise ValueError(f"Unexpected shape {category}: {name}")
    return {
        "filename": name,
        "shape": shape,
        "fold": row["fold"],
        "status": row["Status"],
        "reason": row.get("Reason", ""),
        "mask": mask,
        "algorithm_sha256": row["Algorithm_SHA256"],
        "run_id": row["Run_ID"],
        "loop": parse_line(row, "loop"),
        "thin": branches[0],
        "wide": branches[1],
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    family = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{family}", size)
    except OSError:
        return ImageFont.load_default()


def write(draw: ImageDraw.ImageDraw, xy: tuple[int, int], content: str,
          size: int = 21, color: tuple[int, int, int] = WHITE,
          bold: bool = False) -> None:
    draw.text(xy, content, fill=color, font=font(size, bold))


def point_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float],
                caption: str, color: tuple[int, int, int],
                dx: int = 15, dy: int = -38) -> None:
    px = MASK_LEFT + xy[0] * SCALE
    py = MASK_TOP + xy[1] * SCALE
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color,
                 outline=(8, 12, 18), width=3)
    chosen_font = font(18, True)
    bbox = draw.textbbox((0, 0), caption, font=chosen_font)
    width = bbox[2] - bbox[0] + 16
    height = bbox[3] - bbox[1] + 12
    tx = min(max(int(px) + dx, MASK_LEFT + 5),
             MASK_LEFT + MASK_DISPLAY - width - 5)
    ty = min(max(int(py) + dy, MASK_TOP + 5),
             MASK_TOP + MASK_DISPLAY - height - 5)
    draw.rounded_rectangle((tx, ty, tx + width, ty + height), radius=5,
                           fill=(14, 19, 27), outline=color, width=2)
    draw.text((tx + 8, ty + 5 - bbox[1]), caption, font=chosen_font, fill=color)


def draw_measurement(draw: ImageDraw.ImageDraw, record: dict, key: str) -> None:
    measurement = record[key]
    x1, y1, x2, y2 = measurement["points"]
    color = COLORS[key]
    xy = ((MASK_LEFT + x1 * SCALE, MASK_TOP + y1 * SCALE),
          (MASK_LEFT + x2 * SCALE, MASK_TOP + y2 * SCALE))
    draw.line(xy, fill=(8, 11, 16), width=11)
    draw.line(xy, fill=color, width=6)
    labels = ("U", "D") if key == "loop" else (f"{key} 1", f"{key} 2")
    point_label(draw, (x1, y1), labels[0], color)
    point_label(draw, (x2, y2), labels[1], color, dy=12)


def detail_block(draw: ImageDraw.ImageDraw, record: dict,
                 key: str, title: str, top: int) -> None:
    measurement = record[key]
    color = COLORS[key]
    x1, y1, x2, y2 = measurement["points"]
    draw.rounded_rectangle((INFO_LEFT - 8, top - 10, CANVAS_WIDTH - 35, top + 170),
                           radius=10, fill=(35, 42, 55))
    write(draw, (INFO_LEFT + 9, top), title, 26, color, True)
    write(draw, (INFO_LEFT + 9, top + 38),
          f"Length: {measurement['length']:.2f} px", 26, WHITE, True)
    labels = ("U", "D") if key == "loop" else ("Point 1", "Point 2")
    write(draw, (INFO_LEFT + 9, top + 83),
          f"{labels[0]}: ({x1:.2f}, {y1:.2f})", 19)
    write(draw, (INFO_LEFT + 9, top + 116),
          f"{labels[1]}: ({x2:.2f}, {y2:.2f})", 19)
    if measurement["fallback"]:
        write(draw, (INFO_LEFT + 9, top + 144),
              "EDT fallback (display line is approximate)",
              14, (255, 185, 92))


def render(record: dict, destination: Path) -> None:
    with Image.open(record["mask"]) as source:
        gt = source.convert("L")
    gt = ImageOps.colorize(gt, black=(22, 29, 39),
                           white=(184, 193, 207))
    gt = gt.resize((MASK_DISPLAY, MASK_DISPLAY), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG)
    canvas.paste(gt, (MASK_LEFT, MASK_TOP))
    draw = ImageDraw.Draw(canvas)
    write(draw, (35, 20), record["filename"], 32, WHITE, True)
    write(draw, (36, 62),
          f"GT mask | {record['shape']} | fold {record['fold']} | QC {record['status']}",
          20, GRAY)
    # Wide first; the loop and its U/D points remain foremost near the apex.
    for key in ("wide", "thin", "loop"):
        draw_measurement(draw, record, key)
    write(draw, (INFO_LEFT, 111), "Automatic measurements", 29, WHITE, True)
    write(draw, (INFO_LEFT, 153), "Coordinates are original 512 x 512 pixels", 17, GRAY)
    detail_block(draw, record, "loop", "Loop: predicted U to D", 225)
    detail_block(draw, record, "thin", "Thin branch: GT mask", 432)
    detail_block(draw, record, "wide", "Wide branch: GT mask", 639)
    write(draw, (INFO_LEFT, 857), "Line endpoints = colored dots", 18, GRAY)
    write(draw, (INFO_LEFT, 886), "Thin/wide sorted by measured width", 18, GRAY)
    write(draw, (INFO_LEFT, 915), "Arterial/venous identity not confirmed", 18, GRAY)
    if record["status"] == "Uncertain":
        write(draw, (INFO_LEFT, 963), "QC: Uncertain", 20,
              (255, 185, 92), True)
        reason = record["reason"]
        for i in range(0, min(len(reason), 138), 43):
            write(draw, (INFO_LEFT, 993 + (i // 43) * 25),
                  reason[i:i + 43], 15, GRAY)
    draw.line((35, 1151, CANVAS_WIDTH - 35, 1151),
              fill=(74, 86, 103), width=1)
    write(draw, (36, 1160),
          "Saved GT-analysis endpoints are shown directly; this view does not rerun the algorithm.",
          16, GRAY)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def write_gallery(records: list[dict], destination: Path) -> None:
    cards = []
    for record in records:
        relative = f"{record['shape']}/{Path(record['filename']).stem}_auto.png"
        safe = html.escape(relative, quote=True)
        name = html.escape(record["filename"])
        cards.append(
            f'<a class="card" href="{safe}"><img src="{safe}" loading="lazy" '
            f'alt="{name}"><b>{name}</b><span>'
            f'Loop {record["loop"]["length"]:.1f} px | '
            f'Thin {record["thin"]["length"]:.1f} px | '
            f'Wide {record["wide"]["length"]:.1f} px</span></a>')
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>SELECTED_COUNT GT automatic diameter measurements</title><style>
body{font:16px Arial,sans-serif;background:#151a23;color:#edf2f8;margin:28px}
h1{margin-bottom:8px}p{color:#aab8c8}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:18px}
.card{background:#252d3a;color:#f0f5fc;text-decoration:none;padding:10px;border-radius:8px}
.card img{width:100%;display:block}.card b,.card span{display:block;margin:8px 0 2px}.card span{font-size:13px;color:#aab8c8}
</style><h1>GT mask: automatic diameter measurements</h1>
<p>SELECTED_COUNT images. Select a thumbnail to inspect the exact lines, endpoint coordinates, and pixel lengths.</p>
<div class="grid">""" + "\n".join(cards) + "</div></html>"
    page = page.replace("SELECTED_COUNT", str(len(records)))
    destination.write_text(page, encoding="utf-8")


def output_path(requested: Path | None, selected_n: int = 141) -> Path:
    if requested is not None:
        destination = requested.resolve()
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        return destination
    default_output = DEFAULT_OUTPUT.with_name(f"algorithm_measurements_{selected_n}")
    destination = default_output.resolve()
    number = 2
    while destination.exists():
        destination = default_output.with_name(
            f"{default_output.name}_run{number}").resolve()
        number += 1
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--filename", help="Render only this PNG filename")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    source_metadata = json.loads((results_dir / "provenance.json").read_text(encoding="utf-8-sig"))
    if source_metadata.get("encoder_name") != "dinov3":
        raise ValueError("Expected DINOv3 GT-mask evaluation results")
    selected_n = source_metadata.get("selected_n")
    if type(selected_n) is not int or selected_n <= 0:
        raise ValueError("Evaluation provenance must contain a positive integer selected_n")
    source_csv = results_dir / f"gt_measurements_{selected_n}.csv"
    rows = read_rows(source_csv)
    if len(rows) != selected_n:
        raise ValueError(f"Expected {selected_n} GT analysis results, got {len(rows)}")
    if any(row["Run_ID"] != source_metadata["run_id"] or
           row["Algorithm_SHA256"] != source_metadata["algorithm_sha256"]
           for row in rows.values()):
        raise ValueError("Measurements and evaluation provenance disagree")
    names = [args.filename] if args.filename else sorted(rows)
    if any(name not in rows for name in names):
        raise ValueError(f"Filename not present in GT export: {args.filename}")
    records = [parse_record(rows[name]) for name in names]
    destination = output_path(args.output_dir, selected_n)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}_stage_",
                                     dir=destination.parent) as temporary:
        stage = Path(temporary)
        index_rows = []
        for record in records:
            relative = f"{record['shape']}/{Path(record['filename']).stem}_auto.png"
            render(record, stage / relative)
            index_rows.append({
                "Filename": record["filename"], "shape": record["shape"],
                "fold": record["fold"], "QC": record["status"],
                "loop_px": record["loop"]["length"],
                "thin_px": record["thin"]["length"],
                "wide_px": record["wide"]["length"],
                "GT_mask": str(record["mask"]), "overlay": relative,
            })
        with (stage / "measurements_index.csv").open(
                "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)
        write_gallery(records, stage / "index.html")
        (stage / "provenance.json").write_text(json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_csv": str(source_csv), "source_sha256": sha256(source_csv),
            "algorithm_sha256": sorted({record["algorithm_sha256"]
                                         for record in records}),
            "image_count": len(records),
            "selected_n": selected_n,
            "encoder_name": "dinov3",
            "run_id": source_metadata["run_id"],
            "source": "Saved GT evaluation coordinates and mask files",
            "loop": "DINOv3 MTL predicted U/D straight-line pixel distance",
            "branches": "Limb algorithm diameter lines on GT masks; thin/wide by width",
            "analyzer_rerun": False,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Output appeared during rendering: {destination}")
        # TemporaryDirectory has owner-only ACLs on Windows. Copying its
        # contents into a fresh output folder gives the images the workspace
        # ACL, so the user and the Codex image viewer can open them.
        shutil.copytree(stage, destination)
    print(f"Created {len(records)} GT-mask automatic measurement images: {destination}")
    print(f"Open gallery: {destination / 'index.html'}")


if __name__ == "__main__":
    main()
