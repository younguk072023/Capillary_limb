'''
알고리즘과 전문가와의 신뢰도 평가 코드 
'''
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
        print("CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 1. 데이터 로드
    df_man = pd.read_csv(MANUAL_CSV)
    df_alg = pd.read_csv(ALGO_CSV)

    # 2. 파일명에서 확장자 제거 
    df_man['pure_name'] = df_man['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
    df_alg['pure_name'] = df_alg['Filename'].apply(lambda x: os.path.splitext(str(x))[0])

    # 3. Status 앞뒤 공백 제거 및 Success 데이터만 필터링
    df_alg['Status'] = df_alg['Status'].str.strip()
    df_alg_success = df_alg[df_alg['Status'] == 'Success']

    df = pd.merge(df_man, df_alg_success, on="pure_name")
    
    print("-" * 50)
    print(f"📊 매칭 결과 요약")
    print(f" - 수동 측정 데이터: {len(df_man)}건")
    print(f" - 알고리즘 성공 데이터: {len(df_alg_success)}건")
    print(f" - 최종 매칭 성공(교집합): {len(df)}건")
    print("-" * 50)

    if len(df) == 0:
        print("여전히 매칭되는 데이터가 없습니다!")
        return

    # 5. 데이터 추출 (Arterial + Venous)
    man_vals = np.concatenate([df['Manual_Arterial_Diameter(px)'].values, df['Manual_Venous_Diameter(px)'].values])
    alg_vals = np.concatenate([df['Algo_Arterial_Diameter(px)'].values, df['Algo_Venous_Diameter(px)'].values])

    # 6. 통계 지표 계산
    r_val, p_val = pearsonr(man_vals, alg_vals)
    bias = np.mean(alg_vals - man_vals)
    sd = np.std(alg_vals - man_vals)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    plt.rcParams['font.family'] = 'serif'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Professional vs Proposed", fontsize=16, fontweight='bold')


    ax1.scatter(man_vals, alg_vals, alpha=0.5, color='black', edgecolors='black', s=20, label='Samples')
    
    lims = [min(man_vals.min(), alg_vals.min()) - 5, max(man_vals.max(), alg_vals.max()) + 5]
    ax1.plot(lims, lims, 'r-', linewidth=2, label='y=x')
    
    ax1.set_title(f'Pearson Correlation  (r={r_val:.3f})', fontsize=13)
    ax1.set_xlabel('Professional (px)', fontsize=11)
    ax1.set_ylabel('Proposed (px)', fontsize=11)
    ax1.legend(loc='upper left', frameon=True, edgecolor='black')
    ax1.grid(True, linestyle=':', alpha=0.6)

    
    diff = alg_vals - man_vals
    avg = (alg_vals + man_vals) / 2

    # 파란색 배경 음영 (Shading between LoA)
    ax2.fill_between([avg.min() - 10, avg.max() + 10], lower_loa, upper_loa, color='blue', alpha=0.05)

    # 산점도 (검은색 반투명)
    ax2.scatter(avg, diff, alpha=0.5, color='black', edgecolors='black', s=20, label='Samples')

    # Bias 및 LoA 선 긋기
    ax2.axhline(bias, color='red', linewidth=2, label=f'Bias: {bias:.2f} px')
    ax2.axhline(upper_loa, color='blue', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {upper_loa:.2f} px')
    ax2.axhline(lower_loa, color='blue', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {lower_loa:.2f} px')

    # 그래프 오른쪽 바깥에 수치 텍스트 표시
    text_x = 1.01 # 그래프 오른쪽 경계선 기준 살짝 바깥쪽
    ax2.text(text_x, bias, f'{bias:.2f}', color='red', va='center', fontweight='bold', transform=ax2.get_yaxis_transform())
    ax2.text(text_x, upper_loa, f'{upper_loa:.2f}', color='blue', va='center', transform=ax2.get_yaxis_transform())
    ax2.text(text_x, lower_loa, f'{lower_loa:.2f}', color='blue', va='center', transform=ax2.get_yaxis_transform())

    # 축 및 레이블 설정
    ax2.set_xlim(avg.min() - 2, avg.max() + 2)
    ax2.set_title('Bland-Altman', fontsize=13)
    ax2.set_xlabel('Mean of Diameter (px)', fontsize=11)
    ax2.set_ylabel('Difference (Proposed - Professional) (px)', fontsize=11)
    
    # 범례 설정 (테두리 포함)
    ax2.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 오른쪽 수치 텍스트가 잘리지 않도록 여백 조정
    plt.subplots_adjust(right=0.95)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()