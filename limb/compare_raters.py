"""Compare the two manual raters on the fixed 141-image cohort.

Uses recorded pixel diameters, matched by Filename. No model predictions or
segmentation masks enter this analysis. Total uses all individual pairs.
Bland-Altman differences are Rater 1 minus Rater 2. Plots use micrometers,
with the user-specified calibration of 0.55 micrometers per pixel.
Title-free individual figures are saved as 600-dpi TIFF and 300-dpi PNG files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

DESKTOP = Path.home() / "Desktop"
LIMB = DESKTOP / "Capillary_limb"
WORKSPACE = DESKTOP / "Multi-Task-Learning"
DEFAULT_SELECTION = LIMB / "check" / "diameter_selection_141.csv"
DEFAULT_RATER1 = LIMB / "label" / "outputs" / "section4_review_pa_141" / "manual_review_141.csv"
DEFAULT_RATER2 = LIMB / "label" / "outputs" / "section4_review_pa_141_rater2" / "manual_review_141_rater2.csv"
SHAPES = (("u", "U"), ("tortuous", "Tortuous"), ("crossing", "Crossing"), ("all", "Total"))
MEASURES = (("loop", "루프", "Loop diameter", "manual_loop_px"),
            ("thin_branch", "가는 분지", "Thin branch diameter", "manual_thin_px"),
            ("wide_branch", "굵은 분지", "Wide branch diameter", "manual_wide_px"))
DIFFERENCE_DIRECTION = "Rater 1 minus Rater 2"
DEFAULT_UM_PER_PIXEL = 0.55
SHAPE_COLORS = {"u": "#2474a6", "tortuous": "#c66b20", "crossing": "#418652"}

os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / "outputs" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(LIMB / "check"))
import section4_validation as validation


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", dtype=str, keep_default_na=False)


def shapes(values: pd.Series) -> pd.Series:
    return values.str.strip().str.lower().replace({"twist": "crossing"})


def unique(table: pd.DataFrame, path: Path) -> None:
    if "Filename" not in table or table.Filename.eq("").any() or table.Filename.duplicated().any():
        raise ValueError(f"Missing or duplicate filenames: {path}")


def read_rater(path: Path, selected: pd.DataFrame, sources: dict[Path, bytes]) -> pd.DataFrame:
    sources[path] = path.read_bytes()
    raw = frame(sources[path])
    unique(raw, path)
    if set(raw.Filename) != set(selected.index):
        raise ValueError(f"Rater filenames differ from the fixed cohort: {path}")
    table = raw.set_index("Filename").loc[selected.index].copy()
    if not shapes(table["shape"]).equals(selected["shape"]):
        raise ValueError(f"Rater/cohort shape mismatch: {path}")
    if not table.patient_id.equals(selected.patient_id):
        raise ValueError(f"Rater/cohort patient mismatch: {path}")
    if not table.status.eq("approved").all():
        raise ValueError(f"Incomplete manual measurements: {path}; {table.status.value_counts().to_dict()}")
    numeric = [*validation.POINT_COLUMNS, *validation.MEASURE_COLUMNS, "B1_px", "B2_px"]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(table[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite manual values: {path}")
    if not (table[[*validation.MEASURE_COLUMNS, "B1_px", "B2_px"]] > 0).all().all():
        raise ValueError(f"Non-positive manual diameters: {path}")
    b1 = np.hypot(table.B1_x2 - table.B1_x1, table.B1_y2 - table.B1_y1)
    b2 = np.hypot(table.B2_x2 - table.B2_x1, table.B2_y2 - table.B2_y1)
    loop = np.hypot(table.D_x - table.U_x, table.D_y - table.U_y)
    # GUI stores coordinates to 2 decimals and diameters to 4 decimals.
    for column, calculated in (("manual_loop_px", loop), ("B1_px", b1), ("B2_px", b2)):
        if not np.allclose(table[column], calculated, rtol=0, atol=0.02):
            raise ValueError(f"Saved diameter disagrees with endpoints: {path}, {column}")
    if not (np.allclose(table.manual_thin_px, np.minimum(table.B1_px, table.B2_px), atol=0.0001, rtol=0)
            and np.allclose(table.manual_wide_px, np.maximum(table.B1_px, table.B2_px), atol=0.0001, rtol=0)):
        raise ValueError(f"Thin/wide labels disagree with recorded branch widths: {path}")
    for shape, _ in SHAPES[:-1]:
        split = path.with_name(f"{path.stem}_{shape}.csv")
        sources[split] = split.read_bytes()
        split_table = frame(sources[split])
        unique(split_table, split)
        expected = raw.loc[shapes(raw["shape"]).eq(shape)].set_index("Filename").sort_index()
        actual = split_table.set_index("Filename").sort_index()
        if not actual.equals(expected):
            raise ValueError(f"Type-specific CSV differs from the combined CSV: {split}")
    return table


def summarize(pairs: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    for shape, label in SHAPES:
        part = pairs if shape == "all" else pairs.loc[pairs["shape"].eq(shape)]
        for key, korean, _, source_column in MEASURES:
            a = part[f"rater1_{source_column}"].to_numpy(dtype=float)
            b = part[f"rater2_{source_column}"].to_numpy(dtype=float)
            # The shared helper subtracts its first array from its second.
            values = validation._point_metrics(b, a)
            if not np.isfinite(list(values.values())).all():
                raise ValueError(f"Undefined agreement statistic: {shape}/{key}")
            rows.append({"shape": shape, "shape_label": label, "measure": key,
                         "measure_label": korean, "n": len(part),
                         "patient_n": int(part.patient_id.nunique()),
                         "rater1_mean_px": a.mean(), "rater1_sd_px": a.std(ddof=1),
                         "rater2_mean_px": b.mean(), "rater2_sd_px": b.std(ddof=1),
                         "difference_direction": DIFFERENCE_DIRECTION,
                         "difference_sd_px": (a - b).std(ddof=1),
                         **values, **validation._patient_bootstrap(part, b, a, repetitions)})
    return pd.DataFrame(rows)


def add_micrometer_columns(table: pd.DataFrame, um_per_pixel: float) -> pd.DataFrame:
    """Keep recorded pixels and add explicitly named micrometer lengths/intervals."""
    converted = {}
    for column in table:
        if column.endswith("_px") or "_px_ci95_" in column:
            converted[column.replace("_px", "_um")] = table[column].to_numpy(dtype=float) * um_per_pixel
    return pd.concat([table, pd.DataFrame(converted, index=table.index)], axis=1)


def draw_scatter_panel(ax, pairs: pd.DataFrame, source_column: str) -> None:
    column = source_column.replace("_px", "_um")
    first, second = pairs[f"rater1_{column}"], pairs[f"rater2_{column}"]
    for shape, label in SHAPES[:-1]:
        part = pairs["shape"].eq(shape)
        ax.scatter(first[part], second[part], s=22, alpha=0.75,
                   color=SHAPE_COLORS[shape], label=label)
    lower, upper = min(first.min(), second.min()), max(first.max(), second.max())
    margin = max((upper - lower) * 0.06, 0.5)
    limits = (lower - margin, upper + margin)
    ax.plot(limits, limits, "--", color="#666666", linewidth=1.2)
    ax.set(xlim=limits, ylim=limits,
           xlabel="Rater 1 diameter (µm)", ylabel="Rater 2 diameter (µm)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)


def draw_bland_altman_panel(ax, pairs: pd.DataFrame, row: pd.Series,
                           source_column: str) -> None:
    column = source_column.replace("_px", "_um")
    first, second = pairs[f"rater1_{column}"], pairs[f"rater2_{column}"]
    mean, difference = (first + second) / 2, first - second
    bias, sd = difference.mean(), difference.std(ddof=1)
    if not np.allclose([bias, bias - 1.96 * sd, bias + 1.96 * sd],
                       row[["bias_um", "loa_lower_um", "loa_upper_um"]].to_numpy(dtype=float),
                       atol=1e-10, rtol=0):
        raise ValueError("Plot data and micrometer summary disagree")
    for shape, _ in SHAPES[:-1]:
        part = pairs["shape"].eq(shape)
        ax.scatter(mean[part], difference[part], s=22, alpha=0.75,
                   color=SHAPE_COLORS[shape])
    for column_name, label, style in (("bias_um", "Mean", "-"),
                                      ("loa_lower_um", "Lower 95% LoA", "--"),
                                      ("loa_upper_um", "Upper 95% LoA", "--")):
        value = float(row[column_name])
        ax.axhline(value, color="#944444", linestyle=style, linewidth=1.2,
                   label=f"{label}: {value:.2f} µm")
    ax.set(xlabel="Mean of the two raters (µm)",
           ylabel="Rater 1 − Rater 2 (µm)")
    ax.legend(loc="best", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)


def save_figure(fig, stem: Path, tiff: bool = True) -> list[str]:
    """PNG for preview; losslessly compressed 600-dpi TIFF for individual figures."""
    files = []
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=300, facecolor="white")
    files.append(png.name)
    if tiff:
        destination = stem.with_suffix(".tiff")
        fig.savefig(destination, dpi=600, facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        files.append(destination.name)
    plt.close(fig)
    return files


def draw(pairs: pd.DataFrame, statistics: pd.DataFrame, output: Path) -> list[str]:
    """Save each site as a pair and as separate scatter/BA figures, without titles."""
    files = []
    style = {"font.family": "sans-serif", "font.size": 11,
             "axes.titlesize": 11, "axes.labelsize": 11}
    with plt.rc_context(style):
        # A compact overview preserves the user's original 3-by-2 arrangement.
        overview, overview_axes = plt.subplots(3, 2, figsize=(11.5, 13.0), constrained_layout=True)
        for index, (key, _, _, column) in enumerate(MEASURES):
            row = statistics.loc[statistics["shape"].eq("all") & statistics.measure.eq(key)].iloc[0]
            draw_scatter_panel(overview_axes[index, 0], pairs, column)
            draw_bland_altman_panel(overview_axes[index, 1], pairs, row, column)
            pair, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
            draw_scatter_panel(axes[0], pairs, column)
            draw_bland_altman_panel(axes[1], pairs, row, column)
            files.extend(save_figure(pair, output / f"interrater_{key}_scatter_bland_altman_um"))
            scatter, ax = plt.subplots(figsize=(5.4, 5.1), constrained_layout=True)
            draw_scatter_panel(ax, pairs, column)
            files.extend(save_figure(scatter, output / f"interrater_{key}_scatter_um"))
            bland, ax = plt.subplots(figsize=(5.8, 5.1), constrained_layout=True)
            draw_bland_altman_panel(ax, pairs, row, column)
            files.extend(save_figure(bland, output / f"interrater_{key}_bland_altman_um"))
        files.extend(save_figure(overview, output / "interrater_scatter_bland_altman_um", tiff=False))
    return files



def execute(args: argparse.Namespace) -> Path | None:
    sources: dict[Path, bytes] = {}
    selection_path = args.selection_csv.resolve()
    sources[selection_path] = selection_path.read_bytes()
    selected = frame(sources[selection_path])
    unique(selected, selection_path)
    selected = selected.set_index("Filename")
    selected["shape"] = shapes(selected.Category)
    if len(selected) != 141 or selected["shape"].value_counts().to_dict() != {key: 47 for key, _ in SHAPES[:-1]}:
        raise ValueError("Expected exactly 141 images: U 47, Tortuous 47, Crossing 47")
    selected["patient_id"] = selected.index.to_series().str.extract(r"^(p\d+)_", expand=False)
    if selected.patient_id.isna().any():
        raise ValueError("Cannot identify a patient from the filename")
    raters = [read_rater(path.resolve(), selected, sources) for path in (args.rater1_csv, args.rater2_csv)]
    pairs = selected[["shape", "patient_id"]].copy()
    for index, table in enumerate(raters, start=1):
        pairs = pairs.join(table.drop(columns=["shape", "patient_id"]).add_prefix(f"rater{index}_"))
    pairs = pairs.reset_index()
    for key, _, _, column in MEASURES:
        a, b = pairs[f"rater1_{column}"], pairs[f"rater2_{column}"]
        pairs[f"{key}_mean_px"] = (a + b) / 2
        pairs[f"{key}_rater1_minus_rater2_px"] = a - b
    pairs = add_micrometer_columns(pairs, args.um_per_pixel)
    print(f"Verified two completed raters: {len(pairs)} images, {pairs.patient_id.nunique()} patients; U/Tortuous/Crossing 47 each.", flush=True)
    if args.dry_run:
        return None
    print(f"Figure calibration: 1 px = {args.um_per_pixel:g} µm", flush=True)
    print(f"Bland-Altman difference: {DIFFERENCE_DIRECTION} (µm)", flush=True)
    statistics = summarize(pairs, args.bootstrap_reps)
    statistics = add_micrometer_columns(statistics, args.um_per_pixel)
    report = statistics[["shape_label", "measure_label", "n", "pearson_r", "icc_consistency_C1", "icc_absolute_A1"]].rename(
        columns={"shape_label": "형태", "measure_label": "측정 부위", "n": "N", "pearson_r": "Pearson r", "icc_consistency_C1": "ICC(C,1)", "icc_absolute_A1": "ICC(A,1)"})
    for path, data in sources.items():
        if path.read_bytes() != data:
            raise RuntimeError(f"Measurements changed during analysis; rerun after saving: {path}")
    date = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
    output = args.output_dir.resolve() if args.output_dir else WORKSPACE / "outputs" / f"interrater_measurements_141_{date}"
    if args.output_dir is None:
        base, counter = output, 2
        while output.exists():
            output = base.with_name(f"{base.name}_run{counter}")
            counter += 1
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="interrater_", dir=output.parent) as temp:
        stage = Path(temp)
        report.to_csv(stage / "agreement_table_12.csv", index=False, encoding="utf-8-sig")
        statistics.loc[statistics["shape"].eq("all")].to_csv(stage / "agreement_summary.csv", index=False, encoding="utf-8-sig")
        statistics.loc[~statistics["shape"].eq("all")].to_csv(stage / "agreement_by_shape.csv", index=False, encoding="utf-8-sig")
        pairs.to_csv(stage / "comparison_by_image.csv", index=False, encoding="utf-8-sig")
        archive = stage / "source_snapshots"
        archive.mkdir()
        for path, data in sources.items():
            (archive / path.name).write_bytes(data)
        figure_files = draw(pairs, statistics, stage)
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "selected_n": len(pairs), "patient_n": int(pairs.patient_id.nunique()),
                    "shape_counts": pairs["shape"].value_counts().to_dict(),
                    "sources": [{"path": str(p), "sha256": digest(b)} for p, b in sources.items()],
                    "script": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__).read_bytes()),
                    "statistics_source": str(Path(validation.__file__).resolve()),
                    "statistics_source_sha256": digest(Path(validation.__file__).read_bytes()),
                    "comparison": "Rater 1 versus Rater 2 recorded manual diameters, matched by Filename; raw pixels retained with calibrated micrometer columns",
                    "calibration": {"um_per_pixel": args.um_per_pixel,
                                    "source": "User-specified calibration", "raw_unit": "px", "figure_unit": "µm",
                                    "conversion": "diameter, mean, difference, SD, bias, LoA, MAE and their confidence limits multiplied by um_per_pixel"},
                    "difference_direction": DIFFERENCE_DIRECTION,
                    "bland_altman": {"mean": "(Rater 1 + Rater 2) / 2", "unit": "µm",
                                      "limits": "mean difference +/- 1.96 * sample SD of differences (ddof=1)",
                                      "panels": "Individual sites, separate scatter and Bland-Altman panels, plus 3-by-2 overview", "figures": figure_files,
                                      "titles": False, "statistic_heading": False,
                                      "tiff_dpi": 600, "png_dpi": 300},
                    "icc": "Two-way single-measure ICC(C,1) consistency and ICC(A,1) absolute agreement",
                    "total": "Pooled 141 paired measurements; not an average of shape-specific coefficients",
                    "branch_identity": "Each rater's thinner/wider manual branch, not independently confirmed anatomical identities",
                    "confidence_intervals": {"method": "Patient-cluster bootstrap percentile intervals", "repetitions": args.bootstrap_reps, "seed": 20260928},
                    "coordinate_rounding_check_tolerance_px": 0.02}
        (stage / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        for path, data in sources.items():
            if path.read_bytes() != data:
                raise RuntimeError(f"Source changed before export: {path}")
        shutil.copytree(stage, output)
    # Align Hangul (two display cells) with Latin text in the terminal.
    with pd.option_context("display.unicode.east_asian_width", True):
        print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"Saved: {output}", flush=True)
    for filename in figure_files:
        if filename.endswith(".tiff"):
            print(f"Figure: {output / filename}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater1-csv", type=Path, default=DEFAULT_RATER1)
    parser.add_argument("--rater2-csv", type=Path, default=DEFAULT_RATER2)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--um-per-pixel", type=float, default=DEFAULT_UM_PER_PIXEL,
                        help="Micrometers per pixel for figures and *_um CSV columns (default: 0.55)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be nonnegative")
    if not np.isfinite(args.um_per_pixel) or args.um_per_pixel <= 0:
        parser.error("--um-per-pixel must be a finite positive number")
    execute(args)


if __name__ == "__main__":
    main()
