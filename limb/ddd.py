import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# ================= 파일 경로 설정 =================
MANUAL_CSV = r"C:\Users\park_younguk\Desktop\limb_label\manual_measurement_results.csv"
ALGO_CSV = r"C:\Users\park_younguk\Desktop\limb_mtl\algo_gt_measurement_unified.csv"
# =================================================

def main():
    print("🚀 [확장자 무시 버전] 데이터 병합 및 신뢰도 분석 시작...")
    
    if not os.path.exists(MANUAL_CSV) or not os.path.exists(ALGO_CSV):
        print("❌ CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 1. 데이터 로드
    df_man = pd.read_csv(MANUAL_CSV)
    df_alg = pd.read_csv(ALGO_CSV)

    # 2. [핵심] 파일명에서 확장자 제거 (.png, .jpg 등을 떼고 순수 이름만 남김)
    # 예: 'image_01.png' -> 'image_01', 'mask_01.tif' -> 'mask_01'
    df_man['pure_name'] = df_man['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
    df_alg['pure_name'] = df_alg['Filename'].apply(lambda x: os.path.splitext(str(x))[0])

    # 3. Status 앞뒤 공백 제거 및 Success 데이터만 필터링
    df_alg['Status'] = df_alg['Status'].str.strip()
    df_alg_success = df_alg[df_alg['Status'] == 'Success']

    # 4. 순수 이름을 기준으로 병합 (Merge)
    df = pd.merge(df_man, df_alg_success, on="pure_name")
    
    print("-" * 50)
    print(f"📊 매칭 결과 요약")
    print(f" - 수동 측정 데이터: {len(df_man)}건")
    print(f" - 알고리즘 성공 데이터: {len(df_alg_success)}건")
    print(f" - 최종 매칭 성공(교집합): {len(df)}건")
    print("-" * 50)

    if len(df) == 0:
        print("❌ 여전히 매칭되는 데이터가 없습니다!")
        print(f"👉 수동 이름 예시: '{df_man['pure_name'].iloc[0]}'")
        print(f"👉 알고리즘 이름 예시: '{df_alg['pure_name'].iloc[0]}'")
        return

    # 5. 데이터 추출 (Arterial + Venous)
    man_vals = np.concatenate([df['Manual_Arterial_Diameter(px)'].values, df['Manual_Venous_Diameter(px)'].values])
    alg_vals = np.concatenate([df['Algo_Arterial_Diameter(px)'].values, df['Algo_Venous_Diameter(px)'].values])

    # 6. 통계 지표 계산
    r_val, p_val = pearsonr(man_vals, alg_vals)
    mae = np.mean(np.abs(man_vals - alg_vals))
    bias = np.mean(alg_vals - man_vals)
    sd = np.std(alg_vals - man_vals)

    # 7. 시각화
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Reliability Analysis: Manual vs Algorithm (N={len(man_vals)})", fontsize=15, fontweight='bold')

    # [좌] Correlation Plot
    plt.subplot(1, 2, 1)
    plt.scatter(man_vals, alg_vals, alpha=0.6, edgecolors='k', color='orange')
    lims = [min(man_vals.min(), alg_vals.min()) - 5, max(man_vals.max(), alg_vals.max()) + 5]
    plt.plot(lims, lims, 'r--', label='Identity Line (y=x)')
    plt.title(f'Correlation Scatter Plot (r={r_val:.3f})', fontsize=13)
    plt.xlabel('Manual (px)')
    plt.ylabel('Algorithm (px)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # [우] Bland-Altman Plot
    plt.subplot(1, 2, 2)
    diff = alg_vals - man_vals
    avg = (alg_vals + man_vals) / 2
    plt.scatter(avg, diff, alpha=0.6, edgecolors='k', color='skyblue')
    plt.axhline(bias, color='red', label=f'Bias: {bias:.2f}')
    plt.axhline(bias + 1.96*sd, color='gray', linestyle='--', label='Upper LoA')
    plt.axhline(bias - 1.96*sd, color='gray', linestyle='--', label='Lower LoA')
    plt.title('Bland-Altman Agreement Plot', fontsize=13)
    plt.xlabel('Mean of Two Methods (px)')
    plt.ylabel('Difference (Algo - Manual) (px)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()