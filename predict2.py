"""
세정맥 / 세동맥 구분 신뢰도 분석 코드
논문용 개별 그래프 저장 최종 버전

반영 사항
1. a,b,c,d 패널 라벨 없음
2. Pearson / Bland-Altman 각각 개별 저장
3. Times New Roman 적용
4. x축 / y축 label bold
5. 예시 이미지처럼 흰 배경 + 회색 격자
6. Bland-Altman 숫자 라벨 표시
   - +1.96 SD: 값
   - Mean: 값
   - -1.96 SD: 값
7. 숫자 라벨 뒤 박스 완전 제거
8. PNG + TIFF 저장
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
import pingouin as pg

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =========================================================
# 경로 설정
# =========================================================
MANUAL_CSV = r"C:\Users\park_younguk\Desktop\analysis\total\image\passivity.csv"
ALGO_CSV   = r"C:\Users\park_younguk\Desktop\effi_corr\Total\algorithm_alpha_1_3.csv"

SAVE_DIR   = r"C:\Users\park_younguk\Desktop\analysis\total\image\reliability_alpha_1_3"

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# Figure Style
# =========================================================
DPI = 600

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 12,

    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.2,

    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# 유틸 함수
# =========================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )
    return df


def make_pure_name(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda x: os.path.splitext(x.strip())[0])


def format_pval(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def calc_icc_31(manual_vals, algo_vals):
    """
    ICC(3,1) 계산
    """
    manual_vals = np.asarray(manual_vals, dtype=float)
    algo_vals = np.asarray(algo_vals, dtype=float)

    icc_data = pd.DataFrame({
        "targets": list(range(len(manual_vals))) * 2,
        "raters": ["Expert"] * len(manual_vals) + ["Algorithm"] * len(algo_vals),
        "scores": np.concatenate([manual_vals, algo_vals])
    })

    icc_results = pg.intraclass_corr(
        data=icc_data,
        targets="targets",
        raters="raters",
        ratings="scores"
    )

    row = icc_results.set_index("Type").loc["ICC3"]
    return float(row["ICC"]), row["CI95%"]


def bland_altman_stats(manual_vals, algo_vals):
    """
    Bland-Altman 통계 계산

    Difference = Algorithm - Expert
    """
    manual_vals = np.asarray(manual_vals, dtype=float)
    algo_vals = np.asarray(algo_vals, dtype=float)

    diff = algo_vals - manual_vals
    avg = (algo_vals + manual_vals) / 2.0

    mean_diff = np.mean(diff)
    sd = np.std(diff, ddof=1)

    upper_loa = mean_diff + 1.96 * sd
    lower_loa = mean_diff - 1.96 * sd

    return avg, diff, mean_diff, lower_loa, upper_loa, sd


def save_paper_figure(fig, save_path_without_ext):
    """
    PNG + TIFF 저장
    """
    png_path = save_path_without_ext + ".png"
    tif_path = save_path_without_ext + ".tif"

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05
    )

    if PIL_AVAILABLE:
        img = Image.open(png_path)
        img.save(tif_path, compression="tiff_lzw")
    else:
        fig.savefig(
            tif_path,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.05
        )

    return png_path, tif_path


# =========================================================
# 축 스타일
# =========================================================
def set_axis_style(ax):
    """
    예시 이미지 스타일:
    - 흰 배경
    - 회색 격자
    - 검은 축 테두리
    """
    ax.set_facecolor("white")

    ax.grid(
        True,
        color="#D0D0D0",
        linestyle="-",
        linewidth=0.9,
        alpha=1.0
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("black")

    ax.tick_params(
        direction="out",
        length=4,
        width=1.0,
        colors="black"
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


# =========================================================
# Pearson Plot
# =========================================================
def plot_correlation(ax, manual_vals, algo_vals):
    """
    Pearson correlation plot
    """
    manual_vals = np.asarray(manual_vals, dtype=float)
    algo_vals = np.asarray(algo_vals, dtype=float)

    line_min = min(manual_vals.min(), algo_vals.min()) * 0.90
    line_max = max(manual_vals.max(), algo_vals.max()) * 1.10

    set_axis_style(ax)

    # 산점도
    ax.scatter(
        manual_vals,
        algo_vals,
        s=38,
        color="royalblue",
        alpha=0.70,
        edgecolor="none",
        zorder=3
    )

    # y = x 기준선
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="black",
        linestyle="--",
        linewidth=1.6,
        zorder=2
    )

    # 회귀선
    m, b = np.polyfit(manual_vals, algo_vals, 1)
    x_fit = np.linspace(line_min, line_max, 200)

    ax.plot(
        x_fit,
        m * x_fit + b,
        color="#8B0000",
        linestyle="-",
        linewidth=2.0,
        zorder=2
    )

    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)

    ax.set_xlabel(
        "GT Diameter (px)",
        fontweight="bold",
        fontname="Times New Roman"
    )

    ax.set_ylabel(
        "Predicted Diameter (px)",
        fontweight="bold",
        fontname="Times New Roman"
    )


# =========================================================
# Bland-Altman Plot
# =========================================================
def plot_bland_altman(ax, manual_vals, algo_vals):
    """
    Bland-Altman plot
    예시 이미지와 동일한 방향:
    - 숫자 라벨 뒤 박스 없음
    - +1.96 SD / Mean / -1.96 SD 표시
    """
    avg, diff, mean_diff, lower_loa, upper_loa, sd = bland_altman_stats(
        manual_vals,
        algo_vals
    )

    set_axis_style(ax)

    # 산점도
    ax.scatter(
        avg,
        diff,
        s=38,
        color="royalblue",
        alpha=0.70,
        edgecolor="none",
        zorder=3
    )

    # +1.96 SD line
    ax.axhline(
        upper_loa,
        color="black",
        linestyle="--",
        linewidth=1.8,
        zorder=2
    )

    # Mean line
    ax.axhline(
        mean_diff,
        color="#8B0000",
        linestyle="-",
        linewidth=2.0,
        zorder=2
    )

    # -1.96 SD line
    ax.axhline(
        lower_loa,
        color="black",
        linestyle="--",
        linewidth=1.8,
        zorder=2
    )

    # zero reference line
    ax.axhline(
        0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        zorder=1
    )

    # x축 범위
    x_min = avg.min()
    x_max = avg.max()
    x_margin = max((x_max - x_min) * 0.08, 1.0)

    ax.set_xlim(
        x_min - x_margin,
        x_max + x_margin
    )

    # y축 범위
    y_min_data = min(diff.min(), lower_loa)
    y_max_data = max(diff.max(), upper_loa)
    y_margin = max((y_max_data - y_min_data) * 0.12, sd * 0.5, 1.0)

    ax.set_ylim(
        y_min_data - y_margin,
        y_max_data + y_margin
    )

    # 텍스트 위치 계산
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()

    x_range = x_right - x_left
    y_range = y_top - y_bottom

    text_x = x_right - x_range * 0.07

    # 예시 이미지처럼 선 바로 위/아래에 위치
    upper_text_y = upper_loa + y_range * 0.018
    mean_text_y  = mean_diff + y_range * 0.018
    lower_text_y = lower_loa - y_range * 0.018

    # =====================================================
    # 중요:
    # bbox 없음.
    # alpha 없음.
    # 즉, 텍스트 뒤에 투명/불투명 박스 자체가 없음.
    # =====================================================

    ax.text(
        text_x,
        upper_text_y,
        f"+1.96 SD: {upper_loa:.2f}",
        fontsize=11,
        fontname="Times New Roman",
        color="black",
        ha="right",
        va="bottom"
    )

    ax.text(
        text_x,
        mean_text_y,
        f"Mean: {mean_diff:.2f}",
        fontsize=11,
        fontname="Times New Roman",
        fontweight="bold",
        color="#8B0000",
        ha="right",
        va="bottom"
    )

    ax.text(
        text_x,
        lower_text_y,
        f"-1.96 SD: {lower_loa:.2f}",
        fontsize=11,
        fontname="Times New Roman",
        color="black",
        ha="right",
        va="top"
    )

    ax.set_xlabel(
        "Mean of Diameter (px)",
        fontweight="bold",
        fontname="Times New Roman"
    )

    ax.set_ylabel(
        "Difference (Predicted - GT) (px)",
        fontweight="bold",
        fontname="Times New Roman"
    )


# =========================================================
# 분석 함수
# =========================================================
def analyze_target(df, manual_col, algo_col, target_name, save_dir):
    """
    하나의 target에 대해:
    - Pearson 계산
    - ICC 계산
    - Bland-Altman 계산
    - Pearson plot 저장
    - Bland-Altman plot 저장
    """
    work = df[[manual_col, algo_col]].dropna().copy()

    manual_vals = work[manual_col].astype(float).values
    algo_vals = work[algo_col].astype(float).values

    n = len(work)

    if n < 3:
        raise ValueError(f"{target_name}: 유효 샘플 수가 너무 적음. n = {n}")

    # Pearson
    r_val, p_val = pearsonr(manual_vals, algo_vals)

    # ICC
    icc_val, icc_ci = calc_icc_31(manual_vals, algo_vals)

    # Bland-Altman
    avg, diff, mean_diff, lower_loa, upper_loa, sd = bland_altman_stats(
        manual_vals,
        algo_vals
    )

    # -------------------------------------------------
    # Pearson 저장
    # -------------------------------------------------
    fig_p, ax_p = plt.subplots(figsize=(6.2, 5.2))

    plot_correlation(
        ax=ax_p,
        manual_vals=manual_vals,
        algo_vals=algo_vals
    )

    fig_p.tight_layout()

    pearson_base = os.path.join(
        save_dir,
        f"{target_name}_Pearson"
    )

    fig_p_png, fig_p_tif = save_paper_figure(
        fig=fig_p,
        save_path_without_ext=pearson_base
    )

    plt.close(fig_p)

    # -------------------------------------------------
    # Bland-Altman 저장
    # -------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(7.2, 5.6))

    plot_bland_altman(
        ax=ax_b,
        manual_vals=manual_vals,
        algo_vals=algo_vals
    )

    fig_b.tight_layout()

    ba_base = os.path.join(
        save_dir,
        f"{target_name}_BlandAltman"
    )

    fig_b_png, fig_b_tif = save_paper_figure(
        fig=fig_b,
        save_path_without_ext=ba_base
    )

    plt.close(fig_b)

    return {
        "Target": target_name,
        "N": n,

        "Pearson r": round(r_val, 4),
        "Pearson p": p_val,
        "Pearson p formatted": format_pval(p_val),

        "ICC(3,1)": round(icc_val, 4),
        "ICC 95% CI": str(icc_ci),

        "Mean difference": round(mean_diff, 4),
        "SD": round(sd, 4),
        "Lower LoA": round(lower_loa, 4),
        "Upper LoA": round(upper_loa, 4),

        "Fig_Pearson_png": fig_p_png,
        "Fig_Pearson_tif": fig_p_tif,
        "Fig_BA_png": fig_b_png,
        "Fig_BA_tif": fig_b_tif
    }


# =========================================================
# 메인
# =========================================================
def main():
    print("[Separate AV Reliability Analysis - Final Example Style] 시작")

    if not os.path.exists(MANUAL_CSV):
        raise FileNotFoundError(f"MANUAL_CSV를 찾을 수 없음: {MANUAL_CSV}")

    if not os.path.exists(ALGO_CSV):
        raise FileNotFoundError(f"ALGO_CSV를 찾을 수 없음: {ALGO_CSV}")

    # -------------------------------------------------
    # 1. CSV 로드
    # -------------------------------------------------
    df_man = clean_columns(pd.read_csv(MANUAL_CSV))
    df_alg = clean_columns(pd.read_csv(ALGO_CSV))

    if "Filename" not in df_man.columns:
        raise KeyError("수동 측정 CSV에 'Filename' 컬럼이 필요함.")

    if "Filename" not in df_alg.columns:
        raise KeyError("알고리즘 측정 CSV에 'Filename' 컬럼이 필요함.")

    df_man["pure_name"] = make_pure_name(df_man["Filename"])
    df_alg["pure_name"] = make_pure_name(df_alg["Filename"])

    # -------------------------------------------------
    # 2. Success만 사용
    # -------------------------------------------------
    if "Status" in df_alg.columns:
        df_alg = df_alg[
            df_alg["Status"].astype(str).str.strip() == "Success"
        ].copy()

    # -------------------------------------------------
    # 3. 병합
    # -------------------------------------------------
    df = pd.merge(
        df_man,
        df_alg,
        on="pure_name",
        suffixes=("_man", "_alg")
    )

    if len(df) == 0:
        raise ValueError("매칭되는 데이터가 없음. Filename 또는 pure_name 확인 필요.")

    print(f"매칭된 샘플 수: {len(df)}")

    # -------------------------------------------------
    # 4. 수동 측정값
    # -------------------------------------------------
    manual_pair1_col = "Manual_Arterial_Diameter(px)"
    manual_pair2_col = "Manual_Venous_Diameter(px)"

    missing_manual = [
        c for c in [manual_pair1_col, manual_pair2_col]
        if c not in df.columns
    ]

    if missing_manual:
        raise KeyError(f"수동 측정 컬럼이 없음: {missing_manual}")

    df["Manual_pair1(px)"] = pd.to_numeric(
        df[manual_pair1_col],
        errors="coerce"
    )

    df["Manual_pair2(px)"] = pd.to_numeric(
        df[manual_pair2_col],
        errors="coerce"
    )

    # 작은 값 = arterial, 큰 값 = venous
    df["Manual_Arterial_Diameter_sorted(px)"] = df[
        ["Manual_pair1(px)", "Manual_pair2(px)"]
    ].min(axis=1)

    df["Manual_Venous_Diameter_sorted(px)"] = df[
        ["Manual_pair1(px)", "Manual_pair2(px)"]
    ].max(axis=1)

    # -------------------------------------------------
    # 5. 알고리즘 측정값 컬럼 자동 탐색
    # -------------------------------------------------
    algo_candidates_1 = [
        "Algo_Arter",
        "Algo_Arterial_Diameter(px)",
        "Algo_Arterial_Diameter",
        "Algo_Left",
        "Algo_pair1(px)",
        "Algo_pair1"
    ]

    algo_candidates_2 = [
        "Algo_Ven",
        "Algo_Venous_Diameter(px)",
        "Algo_Venous_Diameter",
        "Algo_Right",
        "Algo_pair2(px)",
        "Algo_pair2"
    ]

    algo_pair1_col = next(
        (c for c in algo_candidates_1 if c in df.columns),
        None
    )

    algo_pair2_col = next(
        (c for c in algo_candidates_2 if c in df.columns),
        None
    )

    if algo_pair1_col is None:
        raise KeyError(
            "알고리즘 pair1 컬럼을 찾지 못함. "
            f"가능 후보: {algo_candidates_1}"
        )

    if algo_pair2_col is None:
        raise KeyError(
            "알고리즘 pair2 컬럼을 찾지 못함. "
            f"가능 후보: {algo_candidates_2}"
        )

    print(f"사용된 알고리즘 pair1 컬럼: {algo_pair1_col}")
    print(f"사용된 알고리즘 pair2 컬럼: {algo_pair2_col}")

    df["Algo_pair1(px)"] = pd.to_numeric(
        df[algo_pair1_col],
        errors="coerce"
    )

    df["Algo_pair2(px)"] = pd.to_numeric(
        df[algo_pair2_col],
        errors="coerce"
    )

    # 작은 값 = arterial, 큰 값 = venous
    df["Algo_Arterial_Diameter_sorted(px)"] = df[
        ["Algo_pair1(px)", "Algo_pair2(px)"]
    ].min(axis=1)

    df["Algo_Venous_Diameter_sorted(px)"] = df[
        ["Algo_pair1(px)", "Algo_pair2(px)"]
    ].max(axis=1)

    # -------------------------------------------------
    # 6. Arterial / Venous 분석
    # -------------------------------------------------
    results = []

    arterial_result = analyze_target(
        df=df,
        manual_col="Manual_Arterial_Diameter_sorted(px)",
        algo_col="Algo_Arterial_Diameter_sorted(px)",
        target_name="Arterial",
        save_dir=SAVE_DIR
    )

    results.append(arterial_result)

    venous_result = analyze_target(
        df=df,
        manual_col="Manual_Venous_Diameter_sorted(px)",
        algo_col="Algo_Venous_Diameter_sorted(px)",
        target_name="Venous",
        save_dir=SAVE_DIR
    )

    results.append(venous_result)

    # -------------------------------------------------
    # 7. Pooled limb 통계만 계산
    # -------------------------------------------------
    pooled_manual = np.concatenate([
        df["Manual_Arterial_Diameter_sorted(px)"].dropna().values,
        df["Manual_Venous_Diameter_sorted(px)"].dropna().values
    ])

    pooled_algo = np.concatenate([
        df["Algo_Arterial_Diameter_sorted(px)"].dropna().values,
        df["Algo_Venous_Diameter_sorted(px)"].dropna().values
    ])

    if len(pooled_manual) >= 3:
        pooled_r, pooled_p = pearsonr(pooled_manual, pooled_algo)
        pooled_icc, pooled_icc_ci = calc_icc_31(pooled_manual, pooled_algo)

        _, _, pooled_mean_diff, pooled_lower_loa, pooled_upper_loa, pooled_sd = bland_altman_stats(
            pooled_manual,
            pooled_algo
        )

        results.append({
            "Target": "Pooled limb (arterial + venous)",
            "N": len(pooled_manual),

            "Pearson r": round(pooled_r, 4),
            "Pearson p": pooled_p,
            "Pearson p formatted": format_pval(pooled_p),

            "ICC(3,1)": round(pooled_icc, 4),
            "ICC 95% CI": str(pooled_icc_ci),

            "Mean difference": round(pooled_mean_diff, 4),
            "SD": round(pooled_sd, 4),
            "Lower LoA": round(pooled_lower_loa, 4),
            "Upper LoA": round(pooled_upper_loa, 4),

            "Fig_Pearson_png": "-",
            "Fig_Pearson_tif": "-",
            "Fig_BA_png": "-",
            "Fig_BA_tif": "-"
        })

    # -------------------------------------------------
    # 8. 결과 CSV 저장
    # -------------------------------------------------
    result_df = pd.DataFrame(results)

    result_csv = os.path.join(
        SAVE_DIR,
        "Separate_Reliability_Summary_final.csv"
    )

    result_df.to_csv(
        result_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # -------------------------------------------------
    # 9. 샘플별 정렬값 저장
    # -------------------------------------------------
    sample_csv = os.path.join(
        SAVE_DIR,
        "AV_sorted_per_sample_final.csv"
    )

    save_cols = [
        "pure_name",
        "Manual_pair1(px)",
        "Manual_pair2(px)",
        "Manual_Arterial_Diameter_sorted(px)",
        "Manual_Venous_Diameter_sorted(px)",
        "Algo_pair1(px)",
        "Algo_pair2(px)",
        "Algo_Arterial_Diameter_sorted(px)",
        "Algo_Venous_Diameter_sorted(px)"
    ]

    df[save_cols].to_csv(
        sample_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # -------------------------------------------------
    # 10. 출력
    # -------------------------------------------------
    print("\n=== 최종 요약 ===")
    print(
        result_df[
            [
                "Target",
                "N",
                "Pearson r",
                "Pearson p formatted",
                "ICC(3,1)",
                "Mean difference",
                "Lower LoA",
                "Upper LoA"
            ]
        ].to_string(index=False)
    )

    print("\n=== 저장 완료 ===")
    print(f"결과 CSV: {result_csv}")
    print(f"샘플별 CSV: {sample_csv}")
    print(f"저장 폴더: {SAVE_DIR}")

    print("\n저장된 그림:")
    print("- Arterial_Pearson.png / .tif")
    print("- Arterial_BlandAltman.png / .tif")
    print("- Venous_Pearson.png / .tif")
    print("- Venous_BlandAltman.png / .tif")


if __name__ == "__main__":
    main()