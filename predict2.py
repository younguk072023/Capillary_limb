'''
세정맥 세동맥 구분 코드

'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pingouin as pg

# =========================================================
# 경로 설정
# =========================================================
MANUAL_CSV = r"C:\Users\park_younguk\Desktop\analysis\total\image\passivity.csv"
ALGO_CSV   = r"C:\Users\park_younguk\Desktop\effi_corr\Total\effi_algorithm.csv"
SAVE_DIR   = r"C:\Users\park_younguk\Desktop\analysis\total\image\reliability_sep_final"

os.makedirs(SAVE_DIR, exist_ok=True)

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


def calc_icc_31(manual_vals, algo_vals):
    """
    ICC(3,1) 계산
    """
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
    diff = algo_vals - manual_vals
    avg = (algo_vals + manual_vals) / 2.0

    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    lower_loa = bias - 1.96 * sd
    upper_loa = bias + 1.96 * sd

    return avg, diff, bias, lower_loa, upper_loa


def analyze_target(df, manual_col, algo_col, target_name, save_dir):
    """
    하나의 target(Arterial / Venous)에 대해
    Pearson, ICC, Bland-Altman 계산 + 그림 저장
    """
    work = df[[manual_col, algo_col]].dropna().copy()

    manual_vals = work[manual_col].astype(float).values
    algo_vals   = work[algo_col].astype(float).values

    n = len(work)
    if n < 3:
        raise ValueError(f"{target_name}: 유효 샘플 수가 너무 적음 (n={n})")

    # Pearson
    r_val, p_val = pearsonr(manual_vals, algo_vals)

    # ICC
    icc_val, icc_ci = calc_icc_31(manual_vals, algo_vals)

    # Bland-Altman
    avg, diff, bias, lower_loa, upper_loa = bland_altman_stats(manual_vals, algo_vals)

    # -------------------------------------------------
    # Plot 저장
    # -------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    line_min = min(manual_vals.min(), algo_vals.min()) * 0.9
    line_max = max(manual_vals.max(), algo_vals.max()) * 1.1

    # Pearson correlation plot
    ax1.scatter(manual_vals, algo_vals, color="royalblue", alpha=0.7, s=35)
    ax1.plot([line_min, line_max], [line_min, line_max],
             color="darkred", linestyle="--", linewidth=1.5)

    m, b = np.polyfit(manual_vals, algo_vals, 1)
    x_fit = np.linspace(line_min, line_max, 200)
    ax1.plot(x_fit, m * x_fit + b, color="black", linewidth=1.2)

    ax1.set_title(f"{target_name}: Pearson Correlation\n(r = {r_val:.3f}, p = {p_val:.3e})",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Expert measurement (px)")
    ax1.set_ylabel("Algorithm measurement (px)")
    ax1.set_xlim(line_min, line_max)
    ax1.set_ylim(line_min, line_max)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Bland-Altman plot
    ax2.scatter(avg, diff, color="royalblue", alpha=0.7, s=35)
    ax2.axhline(bias, color="darkred", linestyle="-", linewidth=1.8)
    ax2.axhline(upper_loa, color="black", linestyle="--", linewidth=1.2)
    ax2.axhline(lower_loa, color="black", linestyle="--", linewidth=1.2)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    xmin, xmax = ax2.get_xlim()
    x_pos = xmin + 0.76 * (xmax - xmin)
    y_offset = 0.02 * (diff.max() - diff.min() + 1e-6)

    ax2.text(x_pos, upper_loa + y_offset, f"+1.96 SD: {upper_loa:.2f}",
             fontsize=9, ha="left", va="bottom")
    ax2.text(x_pos, bias + y_offset, f"Bias: {bias:.2f}",
             fontsize=9, ha="left", va="bottom", color="darkred", fontweight="bold")
    ax2.text(x_pos, lower_loa - y_offset, f"-1.96 SD: {lower_loa:.2f}",
             fontsize=9, ha="left", va="top")

    ax2.set_title(f"{target_name}: Bland-Altman\nICC(3,1) = {icc_val:.3f}",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean of measurements (px)")
    ax2.set_ylabel("Difference (Algo - Expert) (px)")
    ax2.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"{target_name}_Reliability.png")
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close()

    return {
        "Target": target_name,
        "N": n,
        "Pearson r": round(r_val, 4),
        "Pearson p": p_val,
        "ICC(3,1)": round(icc_val, 4),
        "ICC 95% CI": str(icc_ci),
        "Bias (Algo-Expert)": round(bias, 4),
        "Lower LoA": round(lower_loa, 4),
        "Upper LoA": round(upper_loa, 4),
        "Figure": fig_path
    }


# =========================================================
# 메인
# =========================================================
def main():
    print("[Separate AV Reliability Analysis] 시작...")

    if not os.path.exists(MANUAL_CSV):
        raise FileNotFoundError(f"MANUAL_CSV를 찾을 수 없음: {MANUAL_CSV}")
    if not os.path.exists(ALGO_CSV):
        raise FileNotFoundError(f"ALGO_CSV를 찾을 수 없음: {ALGO_CSV}")

    # -------------------------------------------------
    # 1) CSV 로드
    # -------------------------------------------------
    df_man = clean_columns(pd.read_csv(MANUAL_CSV))
    df_alg = clean_columns(pd.read_csv(ALGO_CSV))

    print("\n[Manual CSV columns]")
    print(df_man.columns.tolist())
    print("\n[Algorithm CSV columns]")
    print(df_alg.columns.tolist())

    # 파일명 컬럼
    if "Filename" not in df_man.columns or "Filename" not in df_alg.columns:
        raise KeyError("두 CSV 모두 'Filename' 컬럼이 필요합니다.")

    df_man["pure_name"] = make_pure_name(df_man["Filename"])
    df_alg["pure_name"] = make_pure_name(df_alg["Filename"])

    # Success만
    if "Status" in df_alg.columns:
        df_alg = df_alg[df_alg["Status"].astype(str).str.strip() == "Success"].copy()

    # 병합
    df = pd.merge(df_man, df_alg, on="pure_name", suffixes=("_man", "_alg"))

    if len(df) == 0:
        raise ValueError("매칭되는 데이터가 없습니다.")

    print(f"\n매칭된 샘플 수: {len(df)}")

    # -------------------------------------------------
    # 2) 수동값 불러오기
    #    네 manual csv는 현재 왼쪽/오른쪽 두 다리 값이라고 가정
    #    -> 더 작은 값 = arterial, 더 큰 값 = venous
    # -------------------------------------------------
    # manual csv 쪽 두 다리 직경 컬럼
    manual_pair1_col = "Manual_Arterial_Diameter(px)"
    manual_pair2_col = "Manual_Venous_Diameter(px)"

    if manual_pair1_col not in df.columns or manual_pair2_col not in df.columns:
        raise KeyError(
            f"수동 직경 컬럼이 없습니다.\n"
            f"필요 컬럼: {manual_pair1_col}, {manual_pair2_col}\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )

    df["Manual_pair1(px)"] = pd.to_numeric(df[manual_pair1_col], errors="coerce")
    df["Manual_pair2(px)"] = pd.to_numeric(df[manual_pair2_col], errors="coerce")

    # smaller = arterial, larger = venous
    df["Manual_Arterial_Diameter_sorted(px)"] = df[["Manual_pair1(px)", "Manual_pair2(px)"]].min(axis=1)
    df["Manual_Venous_Diameter_sorted(px)"]   = df[["Manual_pair1(px)", "Manual_pair2(px)"]].max(axis=1)

    # -------------------------------------------------
    # 3) 알고리즘값 불러오기
    #    algorithm csv 역시 왼쪽/오른쪽 두 다리 값이라고 가정
    #    -> 더 작은 값 = arterial, 더 큰 값 = venous
    # -------------------------------------------------
    # 후보 컬럼명
    algo_candidates_1 = [
        "Algo_Arter", "Algo_Arterial_Diameter(px)", "Algo_Arterial_Diameter",
        "Algo_Left", "Algo_pair1(px)", "Algo_pair1"
    ]
    algo_candidates_2 = [
        "Algo_Ven", "Algo_Venous_Diameter(px)", "Algo_Venous_Diameter",
        "Algo_Right", "Algo_pair2(px)", "Algo_pair2"
    ]

    algo_pair1_col = None
    algo_pair2_col = None

    for c in algo_candidates_1:
        if c in df.columns:
            algo_pair1_col = c
            break
    for c in algo_candidates_2:
        if c in df.columns:
            algo_pair2_col = c
            break

    if algo_pair1_col is None or algo_pair2_col is None:
        raise KeyError(
            f"알고리즘 직경 컬럼을 찾지 못했습니다.\n"
            f"pair1 후보: {algo_candidates_1}\n"
            f"pair2 후보: {algo_candidates_2}\n"
            f"현재 컬럼: {df.columns.tolist()}"
        )

    df["Algo_pair1(px)"] = pd.to_numeric(df[algo_pair1_col], errors="coerce")
    df["Algo_pair2(px)"] = pd.to_numeric(df[algo_pair2_col], errors="coerce")

    # smaller = arterial, larger = venous
    df["Algo_Arterial_Diameter_sorted(px)"] = df[["Algo_pair1(px)", "Algo_pair2(px)"]].min(axis=1)
    df["Algo_Venous_Diameter_sorted(px)"]   = df[["Algo_pair1(px)", "Algo_pair2(px)"]].max(axis=1)

    # -------------------------------------------------
    # 4) 각각 따로 평가
    # -------------------------------------------------
    results = []

    # Arterial
    arterial_result = analyze_target(
        df=df,
        manual_col="Manual_Arterial_Diameter_sorted(px)",
        algo_col="Algo_Arterial_Diameter_sorted(px)",
        target_name="Arterial",
        save_dir=SAVE_DIR
    )
    results.append(arterial_result)

    # Venous
    venous_result = analyze_target(
        df=df,
        manual_col="Manual_Venous_Diameter_sorted(px)",
        algo_col="Algo_Venous_Diameter_sorted(px)",
        target_name="Venous",
        save_dir=SAVE_DIR
    )
    results.append(venous_result)

    # -------------------------------------------------
    # 5) pooled limb (보조용)
    # -------------------------------------------------
    pooled_manual = np.concatenate([
        df["Manual_Arterial_Diameter_sorted(px)"].dropna().values,
        df["Manual_Venous_Diameter_sorted(px)"].dropna().values
    ])
    pooled_algo = np.concatenate([
        df["Algo_Arterial_Diameter_sorted(px)"].dropna().values,
        df["Algo_Venous_Diameter_sorted(px)"].dropna().values
    ])

    pooled_r, pooled_p = pearsonr(pooled_manual, pooled_algo)
    pooled_icc, pooled_icc_ci = calc_icc_31(pooled_manual, pooled_algo)
    _, _, pooled_bias, pooled_lower_loa, pooled_upper_loa = bland_altman_stats(pooled_manual, pooled_algo)

    results.append({
        "Target": "Pooled limb (arterial + venous)",
        "N": len(pooled_manual),
        "Pearson r": round(pooled_r, 4),
        "Pearson p": pooled_p,
        "ICC(3,1)": round(pooled_icc, 4),
        "ICC 95% CI": str(pooled_icc_ci),
        "Bias (Algo-Expert)": round(pooled_bias, 4),
        "Lower LoA": round(pooled_lower_loa, 4),
        "Upper LoA": round(pooled_upper_loa, 4),
        "Figure": "-"
    })

    # -------------------------------------------------
    # 6) 저장
    # -------------------------------------------------
    result_df = pd.DataFrame(results)
    result_csv = os.path.join(SAVE_DIR, "Separate_Reliability_Summary.csv")
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    sample_csv = os.path.join(SAVE_DIR, "AV_sorted_per_sample.csv")
    save_cols = [
        "pure_name",
        "Manual_pair1(px)", "Manual_pair2(px)",
        "Manual_Arterial_Diameter_sorted(px)", "Manual_Venous_Diameter_sorted(px)",
        "Algo_pair1(px)", "Algo_pair2(px)",
        "Algo_Arterial_Diameter_sorted(px)", "Algo_Venous_Diameter_sorted(px)"
    ]
    df[save_cols].to_csv(sample_csv, index=False, encoding="utf-8-sig")

    print("\n=== 최종 요약 ===")
    print(result_df.to_string(index=False))
    print()
    print(f"요약 CSV 저장: {result_csv}")
    print(f"샘플별 정렬 결과 저장: {sample_csv}")
    print(f"그림 저장 폴더: {SAVE_DIR}")


if __name__ == "__main__":
    main()