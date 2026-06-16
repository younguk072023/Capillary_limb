"""
두 CSV를 filename 기준으로 합쳐서 클러스터링용 CSV를 만든다.

입력 1: efficientNet-b2.csv
    - filename, pred_length_px 등 (모델 예측 키포인트/길이)
입력 2: final_mtl_algo_measurement.csv
    - Filename, Algo_Arterial_Diameter(px), Algo_Venous_Diameter(px) 등

출력: clustering_input.csv
    - filename, loop_length, arterial_diameter, venous_diameter
    - 성공 샘플만 (세 값이 모두 유효한 행)
"""
import os
import pandas as pd


# ================= 설정 영역 =================
BASE_DIR = r"C:\Users\park_younguk\Desktop\analysis"
LENGTH_CSV = os.path.join(BASE_DIR, "efficientNet-b2.csv")
MEASURE_CSV = os.path.join(BASE_DIR, "effi_mask", "final_mtl_algo_measurement.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "limb_total.csv")
# =============================================


def normalize_name(x):
    """확장자를 떼서 파일명 비교 기준을 통일."""
    return os.path.splitext(str(x))[0]


def main():
    # ---------- 1) 길이 CSV 로드 ----------
    if not os.path.exists(LENGTH_CSV):
        print(f"길이 CSV를 찾지 못함: {LENGTH_CSV}")
        return
    df_len = pd.read_csv(LENGTH_CSV)
    print(f"[1] 길이 CSV 로드: {len(df_len)} 행")

    # filename / pred_length_px 두 열만 뽑아서 key 정규화
    df_len = df_len[["filename", "pred_length_px"]].copy()
    df_len["key"] = df_len["filename"].apply(normalize_name)
    df_len = df_len.rename(columns={"pred_length_px": "loop_length"})

    # ---------- 2) 직경 CSV 로드 ----------
    if not os.path.exists(MEASURE_CSV):
        print(f"측정 CSV를 찾지 못함: {MEASURE_CSV}")
        return
    df_meas = pd.read_csv(MEASURE_CSV)
    print(f"[2] 측정 CSV 로드: {len(df_meas)} 행")

    # Filename / Algo_Arterial_Diameter(px) / Algo_Venous_Diameter(px) 세 열만
    df_meas = df_meas[[
        "Filename",
        "Algo_Arterial_Diameter(px)",
        "Algo_Venous_Diameter(px)",
        "Status",
    ]].copy()
    df_meas["key"] = df_meas["Filename"].apply(normalize_name)
    df_meas = df_meas.rename(columns={
        "Algo_Arterial_Diameter(px)": "arterial_diameter",
        "Algo_Venous_Diameter(px)": "venous_diameter",
    })

    # ---------- 3) key 기준으로 병합 ----------
    merged = pd.merge(
        df_len[["key", "filename", "loop_length"]],
        df_meas[["key", "arterial_diameter", "venous_diameter", "Status"]],
        on="key",
        how="inner",   # 두 쪽에 다 있는 파일만
    )
    print(f"[3] 병합 완료: {len(merged)} 행")

    # ---------- 4) 클러스터링 가능한 행만 필터링 ----------
    # Status가 Success이고, 세 값이 모두 비어있지 않은 행만 남김
    before = len(merged)
    merged = merged[merged["Status"] == "Success"]
    merged = merged.dropna(subset=["loop_length", "arterial_diameter", "venous_diameter"])
    print(f"[4] 유효 샘플 필터링: {before} -> {len(merged)} 행")

    # ---------- 5) 최종 저장 ----------
    final = merged[[
        "filename",
        "loop_length",
        "arterial_diameter",
        "venous_diameter",
    ]].reset_index(drop=True)

    final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("-" * 60)
    print(f"저장 완료: {OUTPUT_CSV}")
    print(f"샘플 수: {len(final)}")
    print("\n[미리보기]")
    print(final.head(10).to_string(index=False))
    print("\n[요약 통계]")
    print(final[["loop_length", "arterial_diameter", "venous_diameter"]].describe().round(2))


if __name__ == "__main__":
    main()