"""Draw Section 4 measurement QA panels from the selected RGB, OOF mask and CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


def rows_by_filename(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["Filename"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError(f"Duplicate filenames in {path}")
    return by_name


def point(row: dict[str, str], x: str, y: str) -> tuple[float, float] | None:
    if not row.get(x) or not row.get(y):
        return None
    return float(row[x]), float(row[y])


def line(draw: ImageDraw.ImageDraw, row: dict[str, str], columns: tuple[str, str, str, str],
         color: str) -> None:
    start = point(row, columns[0], columns[1])
    end = point(row, columns[2], columns[3])
    if start is None or end is None:
        return
    draw.line((start, end), fill=color, width=3)
    for x, y in (start, end):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="black", width=1)


def mask_overlay(original: Image.Image, mask: Image.Image) -> Image.Image:
    mask = mask.convert("L").point(lambda value: 255 if value else 0)
    fill = Image.new("RGBA", original.size, (0, 215, 255, 0))
    fill.putalpha(mask.point(lambda value: 65 if value else 0))
    result = Image.alpha_composite(original.convert("RGBA"), fill)
    edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
    outline = Image.new("RGBA", original.size, (0, 255, 255, 0))
    outline.putalpha(edge.point(lambda value: 220 if value else 0))
    return Image.alpha_composite(result, outline).convert("RGB")


def image_path(selection_dir: Path, shape: str, filename: str) -> Path:
    for folder in selection_dir.iterdir():
        if folder.is_dir() and folder.name.casefold() == shape.casefold():
            path = folder / filename
            if path.is_file():
                return path
    raise FileNotFoundError(f"Selected RGB missing: {shape}/{filename}")


def render(manual: dict[str, str], predicted: dict[str, str], selection_dir: Path,
           output_dir: Path) -> Path:
    filename = manual["Filename"]
    with Image.open(image_path(selection_dir, manual["shape"], filename)) as source:
        original = source.convert("RGB")
    with Image.open(predicted["Predicted_Mask"]) as source:
        mask = source.convert("L")
    if original.size != mask.size:
        raise ValueError(f"RGB/mask size mismatch for {filename}: {original.size}/{mask.size}")

    panels = [original.copy(), mask_overlay(original, mask), original.copy(),
              mask_overlay(original, mask)]
    line(ImageDraw.Draw(panels[1]), predicted,
         ("Pred_U_x", "Pred_U_y", "Pred_D_x", "Pred_D_y"), "#ff2da2")
    manual_draw = ImageDraw.Draw(panels[2])
    for columns, color in (
        (("U_x", "U_y", "D_x", "D_y"), "#ff2da2"),
        (("B1_x1", "B1_y1", "B1_x2", "B1_y2"), "#22ef47"),
        (("B2_x1", "B2_y1", "B2_x2", "B2_y2"), "#ffe600"),
    ):
        line(manual_draw, manual, columns, color)
    predicted_draw = ImageDraw.Draw(panels[3])
    for columns, color in (
        (("Pred_U_x", "Pred_U_y", "Pred_D_x", "Pred_D_y"), "#ff2da2"),
        (("Branch1_x1", "Branch1_y1", "Branch1_x2", "Branch1_y2"), "#22ef47"),
        (("Branch2_x1", "Branch2_y1", "Branch2_x2", "Branch2_y2"), "#ffe600"),
    ):
        line(predicted_draw, predicted, columns, color)

    width, height = original.size
    header = 38
    canvas = Image.new("RGB", (width * 2, (height + header) * 2), "#20242a")
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    titles = ("Original RGB", "OOF mask + predicted U/D", "Manual: loop, B1, B2",
              f"Algorithm: {predicted['Status']}")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = (index % 2) * width
        y = (index // 2) * (height + header)
        draw.text((x + 9, y + 8), title, fill="white", font=font)
        canvas.paste(panel, (x, y + header))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{manual['shape'].lower()}_{Path(filename).stem}_qc.png"
    canvas.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-csv", type=Path, required=True)
    parser.add_argument("--algorithm-csv", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filenames", nargs="+", required=True)
    args = parser.parse_args()
    manual = rows_by_filename(args.manual_csv)
    predicted = rows_by_filename(args.algorithm_csv)
    for filename in args.filenames:
        if filename not in manual or filename not in predicted:
            raise ValueError(f"Missing manual or predicted row: {filename}")
        print(render(manual[filename], predicted[filename], args.selection_dir,
                     args.output_dir))


if __name__ == "__main__":
    main()
