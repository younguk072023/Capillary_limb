"""Measure Section 4 diameters on the original ROI images.

Legacy branch measurements and loop keypoints are imported as *pending* drafts.
``--blind-loop`` leaves new U/D points empty for an independent Section 4 loop
measurement instead of adjudicating the MTL training keypoint labels. Saved
reviews always resume as saved. The original CSVs and images are never modified.
Use ``--independent`` for six blank points on each image. Image folders may be
flat or contain U/tortuous/twist subfolders. Run ``--check`` for a headless
input check, or run without it to open Tkinter.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ANALYSIS = Path(r"C:\Users\USER\Desktop\analysis")
DEFAULT_IMAGES = ANALYSIS / "total" / "image"
DEFAULT_BRANCHES = DEFAULT_IMAGES / "passivity.csv"
DEFAULT_LOOPS = ANALYSIS / "capillary_keypoint_final.csv"
DEFAULT_SHAPES = ANALYSIS / "image"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "section4_review" / "manual_review.csv"

POINTS = ("U", "D", "B1_1", "B1_2", "B2_1", "B2_2")
COORD_COLUMNS = tuple(
    f"{branch}_{axis}{end}"
    for branch in ("B1", "B2") for end in ("1", "2") for axis in ("x", "y")
) + ("U_x", "U_y", "D_x", "D_y")
CSV_COLUMNS = (
    "Filename", "status", "patient_id", "shape", *COORD_COLUMNS,
    "manual_loop_px", "manual_thin_px", "manual_wide_px", "B1_px", "B2_px",
    "thin_branch", "reason", "reviewer", "reviewed_at",
)


def _number(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite coordinate: {value!r}")
    return number


def _point(row: dict[str, str], prefix: str) -> tuple[float, float] | None:
    x, y = _number(row.get(f"{prefix}_x")), _number(row.get(f"{prefix}_y"))
    if (x is None) != (y is None):
        raise ValueError(f"Incomplete point {prefix}")
    return None if x is None else (x, y)


def _branch_point(row: dict[str, str], branch: str, end: str) -> tuple[float, float] | None:
    x, y = _number(row.get(f"{branch}_x{end}")), _number(row.get(f"{branch}_y{end}"))
    if (x is None) != (y is None):
        raise ValueError(f"Incomplete legacy point {branch}{end}")
    return None if x is None else (x, y)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


@dataclass
class Review:
    filename: str
    patient_id: str
    shape: str
    image_relpath: str = ""
    points: dict[str, tuple[float, float] | None] = field(
        default_factory=lambda: {key: None for key in POINTS}
    )
    status: str = "pending"
    reason: str = ""
    reviewer: str = ""
    reviewed_at: str = ""

    def live_lengths(self) -> dict[str, float | None]:
        """Return displayed lengths even while an annotation is incomplete."""
        pairs = {"loop": ("U", "D"), "B1": ("B1_1", "B1_2"),
                 "B2": ("B2_1", "B2_2")}
        return {name: (_distance(self.points[a], self.points[b])
                       if self.points[a] is not None and self.points[b] is not None
                       else None)
                for name, (a, b) in pairs.items()}

    def update_point(self, name: str, position: tuple[float, float],
                     auto_approve: bool = False) -> None:
        """Replace a point and keep a complete quick review approved."""
        if name not in POINTS:
            raise ValueError(f"Unknown measurement point: {name}")
        self.points[name] = position
        self.reason = ""
        lengths = self.live_lengths()
        complete = all(value is not None and value > 0 for value in lengths.values())
        if auto_approve and complete:
            self.status = "approved"
            self.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        else:
            self.status, self.reviewed_at = "pending", ""

    def metrics(self) -> dict[str, float | str]:
        if self.status != "approved":
            return {}
        if any(self.points[key] is None for key in POINTS):
            raise ValueError(f"{self.filename}: approved review has missing points")
        p = self.points
        b1 = _distance(p["B1_1"], p["B1_2"])
        b2 = _distance(p["B2_1"], p["B2_2"])
        loop = _distance(p["U"], p["D"])
        if min(b1, b2, loop) <= 0:
            raise ValueError(f"{self.filename}: zero-length measurement")
        thin_branch = "B1" if b1 < b2 else "B2" if b2 < b1 else "equal"
        return {
            "manual_loop_px": round(loop, 4),
            "manual_thin_px": round(min(b1, b2), 4),
            "manual_wide_px": round(max(b1, b2), 4),
            "B1_px": round(b1, 4),
            "B2_px": round(b2, 4),
            "thin_branch": thin_branch,
        }

    def validate(self) -> None:
        if self.status not in {"pending", "approved", "unmeasurable"}:
            raise ValueError(f"{self.filename}: invalid status {self.status!r}")
        if self.status == "approved":
            self.metrics()
        if self.status == "unmeasurable" and not self.reason.strip():
            raise ValueError(f"{self.filename}: unmeasurable requires a reason")

    def to_row(self) -> dict[str, str | float]:
        self.validate()
        row: dict[str, str | float] = {
            "Filename": self.filename, "status": self.status,
            "patient_id": self.patient_id, "shape": self.shape,
            "reason": self.reason, "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
        }
        for key in POINTS:
            p = self.points[key]
            if key.startswith("B"):
                branch, end = key.split("_")
                prefix = f"{branch}_x{end}", f"{branch}_y{end}"
            else:
                prefix = f"{key}_x", f"{key}_y"
            row[prefix[0]], row[prefix[1]] = ("", "") if p is None else p
        for key in ("manual_loop_px", "manual_thin_px", "manual_wide_px", "B1_px", "B2_px", "thin_branch"):
            row[key] = ""
        row.update(self.metrics())
        return row


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_by_name(path: Path, column: str) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for row in _csv_rows(path):
        name = row[column]
        if name in found:
            raise ValueError(f"Duplicate {name} in {path}")
        found[name] = row
    return found


def _shape_lookup(shape_root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for shape in ("u", "tortuous", "twist"):
        folder = shape_root / shape
        if not folder.exists():
            continue
        for path in folder.glob("*.png"):
            if path.name in found:
                raise ValueError(f"Image appears in multiple shape groups: {path.name}")
            found[path.name] = shape
    return found


def _image_paths(image_dir: Path) -> dict[str, str]:
    """Map unique image names to paths beneath a flat or grouped image root."""
    paths = list(image_dir.glob("*.png"))
    for shape in ("U", "tortuous", "twist"):
        paths.extend((image_dir / shape).glob("*.png"))
    if not paths:
        raise ValueError(f"No original PNGs in {image_dir}")
    found: dict[str, str] = {}
    order = {"u": 0, "tortuous": 1, "twist": 2}
    def sort_key(path: Path) -> tuple[int, str]:
        relative = path.relative_to(image_dir)
        group = relative.parts[0].lower() if len(relative.parts) > 1 else ""
        return (order.get(group, -1), path.name)

    for path in sorted(paths, key=sort_key):
        if path.name in found:
            raise ValueError(f"Duplicate image filename in {image_dir}: {path.name}")
        found[path.name] = path.relative_to(image_dir).as_posix()
    return found


def load_reviews(
    image_dir: Path = DEFAULT_IMAGES,
    branches_csv: Path = DEFAULT_BRANCHES,
    loops_csv: Path = DEFAULT_LOOPS,
    shape_root: Path = DEFAULT_SHAPES,
    output_csv: Path = DEFAULT_OUTPUT,
    blind_loop: bool = False,
    independent: bool = False,
) -> list[Review]:
    image_paths = _image_paths(image_dir)
    images = list(image_paths)
    branches = {} if independent else _rows_by_name(branches_csv, "Filename")
    loops = {} if blind_loop or independent else _rows_by_name(loops_csv, "filename")
    shapes = _shape_lookup(shape_root)
    saved = _rows_by_name(output_csv, "Filename") if output_csv.exists() else {}
    unknown_saved = set(saved) - set(images)
    if unknown_saved:
        raise ValueError(f"Review CSV contains images no longer in input folder: {sorted(unknown_saved)}")
    result: list[Review] = []
    for name in images:
        match = re.match(r"^(p\d+)_", name)
        patient_id = match.group(1) if match else ""
        relative = Path(image_paths[name])
        shape = relative.parts[0].lower() if len(relative.parts) > 1 else shapes.get(name, "")
        item = Review(name, patient_id, shape, image_paths[name])
        if name in branches:
            old = branches[name]
            for target, source, end in (("B1_1", "A", "1"), ("B1_2", "A", "2"),
                                        ("B2_1", "V", "1"), ("B2_2", "V", "2")):
                item.points[target] = _branch_point(old, source, end)
        if name in loops and not blind_loop:
            old = loops[name]
            item.points["U"] = _point(old, "U")
            item.points["D"] = _point(old, "D")
        if name in saved:
            old = saved[name]
            item.status = old.get("status", "pending") or "pending"
            item.reason = old.get("reason", "")
            item.reviewer = old.get("reviewer", "")
            item.reviewed_at = old.get("reviewed_at", "")
            for key in POINTS:
                if key.startswith("B"):
                    branch, end = key.split("_")
                    x, y = _number(old.get(f"{branch}_x{end}")), _number(old.get(f"{branch}_y{end}"))
                    if (x is None) != (y is None):
                        raise ValueError(f"{name}: incomplete saved point {key}")
                    item.points[key] = None if x is None else (x, y)
                else:
                    item.points[key] = _point(old, key)
            item.validate()
        result.append(item)
    return result


def shape_csv_paths(output_csv: Path) -> dict[str, Path]:
    """Use clear file names while retaining `twist` in source CSV rows."""
    return {shape: output_csv.with_name(f"{output_csv.stem}_{shape}{output_csv.suffix}")
            for shape in ("u", "tortuous", "crossing")}


def _shape_tables(rows: list[dict], output_csv: Path) -> list[tuple[Path, list[dict]]]:
    grouped: dict[str, list[dict]] = {shape: [] for shape in shape_csv_paths(output_csv)}
    for row in rows:
        source_shape = str(row["shape"]).strip().lower()
        shape = "crossing" if source_shape == "twist" else source_shape
        if shape not in grouped:
            raise ValueError(f"Unknown shape for {row['Filename']}: {source_shape!r}")
        grouped[shape].append(row)
    return [(path, grouped[shape])
            for shape, path in shape_csv_paths(output_csv).items()]


def _write_tables(tables: list[tuple[Path, list[dict]]]) -> None:
    """Stage every CSV, then replace all; restore prior files if a replace fails."""
    paths = [path.resolve() for path, _ in tables]
    if len(paths) != len(set(paths)):
        raise ValueError("Combined and shape CSV paths must be different")
    staged: list[tuple[Path, Path]] = []
    replaced: list[Path] = []
    original: dict[Path, bytes | None] = {}
    try:
        for path, rows in tables:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            staged.append((path, temporary))
            original[path] = path.read_bytes() if path.exists() else None
        try:
            for path, temporary in staged:
                os.replace(temporary, path)
                replaced.append(path)
        except OSError:
            for path in reversed(replaced):
                before = original[path]
                if before is None:
                    path.unlink(missing_ok=True)
                else:
                    rollback = path.with_suffix(path.suffix + ".rollback.tmp")
                    try:
                        rollback.write_bytes(before)
                        os.replace(rollback, path)
                    finally:
                        rollback.unlink(missing_ok=True)
            raise
    finally:
        for _path, temporary in staged:
            temporary.unlink(missing_ok=True)


def save_reviews(reviews: list[Review], output_csv: Path,
                 split_by_shape: bool = False) -> None:
    rows = [item.to_row() for item in reviews]
    tables = [(output_csv, rows)]
    if split_by_shape:
        tables.extend(_shape_tables(rows, output_csv))
    _write_tables(tables)


def export_shapes_from_saved(output_csv: Path) -> dict[str, Path]:
    """Create shape CSVs from an existing combined CSV without rewriting it."""
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"Unexpected combined CSV columns: {output_csv}")
        rows = list(reader)
    names = [row["Filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate filenames in {output_csv}")
    _write_tables(_shape_tables(rows, output_csv))
    return shape_csv_paths(output_csv)


class ReviewWindow:
    def __init__(self, reviews: list[Review], image_dir: Path, output_csv: Path,
                 simple: bool = False, split_by_shape: bool = False):
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageStat, ImageTk

        self.tk, self.messagebox = tk, messagebox
        self.Image, self.ImageStat, self.ImageTk = Image, ImageStat, ImageTk
        self.reviews, self.image_dir, self.output_csv = reviews, image_dir, output_csv
        self.simple = simple
        self.split_by_shape = split_by_shape
        self.index = 0
        self.root = tk.Tk()
        self.root.title(f"Manual diameter measurement - {len(reviews)} images" if simple
                        else "Section 4 manual diameter review")
        self.root.geometry("1130x800")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.selection = tk.StringVar(value=POINTS[0])
        self.scale = tk.DoubleVar(value=1.5)
        self.reason = tk.StringVar()
        self.reviewer = tk.StringVar()
        self.info = tk.StringVar()
        self.length_vars = {name: tk.StringVar() for name in ("loop", "B1", "B2")}

        left = tk.Frame(self.root)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, background="#222222", width=780, height=730)
        hbar = tk.Scrollbar(left, orient="horizontal", command=self.canvas.xview)
        vbar = tk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self.click_image)

        right = tk.Frame(self.root, padx=12, pady=10)
        right.pack(side="right", fill="y")
        tk.Label(right, textvariable=self.info, justify="left", wraplength=265).pack(anchor="w")
        if simple:
            tk.Label(right, text="Lengths (px)", font=("Arial", 14, "bold")).pack(
                anchor="w", pady=(20, 6))
            for name, color in (("loop", "#198c26"), ("B1", "#cc1717"),
                                ("B2", "#087e99")):
                tk.Label(right, textvariable=self.length_vars[name],
                         font=("Arial", 13, "bold"), fg=color).pack(
                    anchor="w", pady=3)
            tk.Label(right, text="Select a point, then click the image.",
                     wraplength=265, justify="left").pack(anchor="w", pady=(20, 5))
        else:
            tk.Label(right, text="U/D = loop width; B1/B2 = the two limb widths (arterial/venous side is inferred by width)",
                     wraplength=265, justify="left").pack(anchor="w", pady=(12, 4))
        key_widgets = [self.root, self.canvas]
        for name in POINTS:
            button = tk.Radiobutton(right, text=name, variable=self.selection,
                                    value=name)
            button.pack(anchor="w")
            key_widgets.append(button)
        tk.Label(right, text="Zoom").pack(anchor="w", pady=(10, 0))
        zoom = tk.Scale(right, variable=self.scale, from_=1, to=4,
                        resolution=0.5, orient="horizontal",
                        command=lambda _value: self.redraw())
        zoom.pack(fill="x")
        key_widgets.append(zoom)
        if not simple:
            tk.Label(right, text="Reviewer").pack(anchor="w", pady=(8, 0))
            tk.Entry(right, textvariable=self.reviewer).pack(fill="x")
            tk.Label(right, text="Unmeasurable reason (required)").pack(anchor="w", pady=(8, 0))
            tk.Entry(right, textvariable=self.reason).pack(fill="x")
            tk.Button(right, text="Approve all three diameters", command=self.approve).pack(fill="x", pady=(12, 2))
            tk.Button(right, text="Mark unmeasurable", command=self.unmeasurable).pack(fill="x", pady=2)
            tk.Button(right, text="Save pending", command=self.pending).pack(fill="x", pady=2)
        previous = tk.Button(right, text="Previous", command=lambda: self.navigate(-1))
        previous.pack(fill="x", pady=(18, 2))
        following = tk.Button(right, text="Next", command=lambda: self.navigate(1))
        following.pack(fill="x", pady=2)
        key_widgets.extend((previous, following))
        if simple:
            for widget in key_widgets:
                widget.bind("<Left>", lambda event: self._arrow(event, -1))
                widget.bind("<Right>", lambda event: self._arrow(event, 1))
            self.canvas.configure(takefocus=1)
            self.canvas.focus_set()
        else:
            tk.Button(right, text="Save and close", command=self.close).pack(fill="x", pady=(18, 2))
        self._load_fields()
        self.redraw()

    @property
    def current(self) -> Review:
        return self.reviews[self.index]

    def redraw(self) -> None:
        item = self.current
        image = self.Image.open(self.image_dir / item.image_relpath).convert("RGB")
        self.image_size = image.size
        right_edge = image.crop((image.width - 1, 0, image.width, image.height))
        edge_rgb = self.ImageStat.Stat(right_edge).median
        edge_color = "#%02x%02x%02x" % tuple(round(value) for value in edge_rgb)
        self.canvas.configure(background=edge_color)
        z = self.scale.get()
        view = image.resize((round(image.width * z), round(image.height * z)), self.Image.Resampling.NEAREST)
        self.tk_image = self.ImageTk.PhotoImage(view)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, view.width, view.height))
        for names, color in ((("B1_1", "B1_2"), "#e32929"),
                             (("B2_1", "B2_2"), "#17bbdb"),
                             (("U", "D"), "#46e04e")):
            a, b = (item.points[name] for name in names)
            if a is not None and b is not None:
                self.canvas.create_line(a[0]*z, a[1]*z, b[0]*z, b[1]*z,
                                        fill=color, width=2)
            for name in names:
                p = item.points[name]
                if p is not None:
                    x, y = p[0]*z, p[1]*z
                    self.canvas.create_oval(x-4, y-4, x+4, y+4,
                                            fill=color, outline="white")
                    self.canvas.create_text(x+6, y-7, text=name,
                                            anchor="sw", fill=color, font=("Arial", 10, "bold"))
        if self.simple:
            self.info.set(
                f"{self.index + 1}/{len(self.reviews)}  {item.filename}\n"
                f"Patient {item.patient_id}  Shape {item.shape or '?'}"
            )
            for key, label in (("loop", "Loop U-D"), ("B1", "B1"),
                               ("B2", "B2")):
                value = item.live_lengths()[key]
                self.length_vars[key].set(
                    f"{label}: {value:.2f} px" if value is not None
                    else f"{label}: —")
        else:
            counts = {state: sum(row.status == state for row in self.reviews)
                      for state in ("approved", "unmeasurable", "pending")}
            self.info.set(
                f"{self.index + 1}/{len(self.reviews)}  {item.filename}\n"
                f"Patient {item.patient_id}  Shape {item.shape or '?'}\n"
                f"Status: {item.status}\n"
                f"Approved {counts['approved']} · Unmeasurable {counts['unmeasurable']} · Pending {counts['pending']}"
            )

    def _load_fields(self) -> None:
        self.reason.set(self.current.reason)
        self.reviewer.set(self.current.reviewer)

    def click_image(self, event) -> None:
        if self.simple:
            self.canvas.focus_set()
        z = self.scale.get()
        x, y = self.canvas.canvasx(event.x) / z, self.canvas.canvasy(event.y) / z
        if not (0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]):
            return
        item = self.current
        name = self.selection.get()
        before = (item.points[name], item.status, item.reason, item.reviewed_at)
        item.update_point(name, (round(x, 2), round(y, 2)),
                          auto_approve=self.simple)
        self.reason.set("")
        if self.simple:
            self._save_fields()
            if not self._save():
                item.points[name], item.status, item.reason, item.reviewed_at = before
                self.reason.set(item.reason)
                self.redraw()
                return
        position = POINTS.index(name)
        self.selection.set(POINTS[(position + 1) % len(POINTS)])
        self.redraw()

    def _arrow(self, _event, change: int) -> str:
        self.navigate(change)
        return "break"

    def _save_fields(self) -> None:
        self.current.reason = self.reason.get().strip()
        self.current.reviewer = self.reviewer.get().strip()

    def _save(self) -> bool:
        try:
            save_reviews(self.reviews, self.output_csv,
                         split_by_shape=self.split_by_shape)
            return True
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Could not save review", str(exc))
            return False

    def approve(self) -> None:
        self._save_fields()
        item = self.current
        previous = (item.status, item.reason, item.reviewed_at)
        item.status, item.reason = "approved", ""
        item.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self._save():
            self._load_fields()
            self.redraw()
        else:
            item.status, item.reason, item.reviewed_at = previous

    def unmeasurable(self) -> None:
        self._save_fields()
        item = self.current
        previous = (item.status, item.reason, item.reviewed_at)
        item.status = "unmeasurable"
        item.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self._save():
            self.redraw()
        else:
            item.status, item.reason, item.reviewed_at = previous

    def pending(self) -> None:
        self._save_fields()
        item = self.current
        previous = (item.status, item.reviewed_at)
        item.status, item.reviewed_at = "pending", ""
        if self._save():
            self.redraw()
        else:
            item.status, item.reviewed_at = previous

    def navigate(self, change: int) -> None:
        self._save_fields()
        if not self._save():
            return
        self.index = max(0, min(len(self.reviews) - 1, self.index + change))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._load_fields()
        self.redraw()

    def close(self) -> None:
        self._save_fields()
        if self._save():
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--branches-csv", type=Path, default=DEFAULT_BRANCHES)
    parser.add_argument("--loops-csv", type=Path, default=DEFAULT_LOOPS)
    parser.add_argument("--shape-root", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blind-loop", action="store_true",
                        help="new independent U/D annotation; do not preload MTL training loop keypoints for unsaved images")
    parser.add_argument("--independent", action="store_true",
                        help="start all six points blank; do not import any legacy branch or loop coordinates")
    parser.add_argument("--simple", action="store_true",
                        help="minimal UI with live lengths and automatic approval/save")
    parser.add_argument("--split-by-shape", action="store_true",
                        help="save U, tortuous, and crossing subset CSVs beside the combined CSV")
    parser.add_argument("--check", action="store_true", help="check inputs without opening GUI or writing")
    args = parser.parse_args()
    reviews = load_reviews(args.image_dir, args.branches_csv, args.loops_csv,
                           args.shape_root, args.output_csv, args.blind_loop,
                           args.independent)
    if args.check:
        print(f"Images: {len(reviews)}; all 6 points: "
              f"{sum(all(item.points.values()) for item in reviews)}; "
              f"approved: {sum(item.status == 'approved' for item in reviews)}; "
              f"pending: {sum(item.status == 'pending' for item in reviews)}; "
              f"unmeasurable: {sum(item.status == 'unmeasurable' for item in reviews)}")
        print(f"Output: {args.output_csv}")
        return
    ReviewWindow(reviews, args.image_dir, args.output_csv,
                 simple=args.simple, split_by_shape=args.split_by_shape).run()


if __name__ == "__main__":
    main()
