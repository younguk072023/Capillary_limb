import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pingouin as pg  # ICC 계산을 위한 라이브러리
import os

# 1. 경로 설정
MANUAL_CSV = r"C:\Users\park_younguk\Desktop\analysis\total\image\manual_measurement_results.csv"
ALGO_CSV = r"C:\Users\park_younguk\Desktop\analysis\total\label\algo_gt_measurement_unified.csv"

def main():
    print("🚀 [Publication Style] 통합 신뢰도 분석 (Pearson, Bland-Altman, ICC) 시작...")
    
    if not os.path.exists(MANUAL_CSV) or not os.path.exists(ALGO_CSV):
        print("CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 데이터 로드 및 병합
    df_man = pd.read_csv(MANUAL_CSV)
    df_alg = pd.read_csv(ALGO_CSV)

    df_man['pure_name'] = df_man['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
    df_alg['pure_name'] = df_alg['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
    
    # Success 데이터만 필터링
    df_alg_success = df_alg[df_alg['Status'].str.strip() == 'Success']
    df = pd.merge(df_man, df_alg_success, on="pure_name")
    
    if len(df) == 0:
        print("매칭되는 데이터가 없습니다!")
        return

    # Arterial + Venous 데이터 통합 (픽셀 단위 -> 필요시 여기에 변환계수 곱하기)
    man_vals = np.concatenate([df['Manual_Arterial_Diameter(px)'].values, df['Manual_Venous_Diameter(px)'].values])
    alg_vals = np.concatenate([df['Algo_Arterial_Diameter(px)'].values, df['Algo_Venous_Diameter(px)'].values])

    # 3. 통계 계산 - Pearson & Bland-Altman
    r_val, p_val = pearsonr(man_vals, alg_vals)
    diff = alg_vals - man_vals
    avg = (alg_vals + man_vals) / 2
    bias = np.mean(diff)
    sd = np.std(diff)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    # 4. 통계 계산 - ICC (3,1)
    # ICC 계산을 위한 데이터 재구성 (Long-format)
    icc_data = pd.DataFrame({
        'targets': list(range(len(man_vals))) * 2,
        'raters': ['Expert'] * len(man_vals) + ['Algorithm'] * len(alg_vals),
        'scores': np.concatenate([man_vals, alg_vals])
    })
    icc_results = pg.intraclass_corr(data=icc_data, targets='targets', raters='raters', ratings='scores')
    icc_31 = icc_results.set_index('Type').loc['ICC3', 'ICC']
    icc_ci = icc_results.set_index('Type').loc['ICC3', 'CI95%']

    # 5. 시각화 설정 (Times New Roman 적용)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_style("white")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    marker_style = dict(color='royalblue', alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

    # --- Left: Pearson Correlation Plot ---
    ax1.scatter(man_vals, alg_vals, **marker_style, label='Data points')
    
    # Identity & Regression Line
    line_min = min(man_vals.min(), alg_vals.min()) * 0.9
    line_max = max(man_vals.max(), alg_vals.max()) * 1.1
    ax1.plot([line_min, line_max], [line_min, line_max], color='darkred', linestyle='--',linewidth=1.5, label='Identity (y=x)')
    
    m, b = np.polyfit(man_vals, alg_vals, 1)
    ax1.plot(man_vals, m*man_vals + b, color='black', linewidth=1, label='Regression')

    ax1.set_title(f'Pearson Correlation\n(r = {r_val:.3f}, p < 0.001)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Expert Measurement (px)', fontsize=11)
    ax1.set_ylabel('Proposed Algorithm (px)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(line_min, line_max); ax1.set_ylim(line_min, line_max)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # --- Right: Bland-Altman Plot ---
    ax2.scatter(avg, diff, **marker_style)
    ax2.axhline(bias, color='darkred', linestyle='-', linewidth=2)
    ax2.axhline(upper_loa, color='black', linestyle='--', linewidth=1.2)
    ax2.axhline(lower_loa, color='black', linestyle='--', linewidth=1.2)
    ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8)

    # 텍스트 주석 (오른쪽 끝에 배치)
    x_pos = ax2.get_xlim()[1] * 0.98
    ax2.text(x_pos, bias, f'Bias: {bias:.2f}', va='bottom', ha='right', color='darkred', fontweight='bold')
    ax2.text(x_pos, upper_loa, f'+1.96 SD: {upper_loa:.2f}', va='bottom', ha='right', fontsize=9)
    ax2.text(x_pos, lower_loa, f'-1.96 SD: {lower_loa:.2f}', va='top', ha='right', fontsize=9)

    ax2.set_title(f'Bland-Altman Plot\n(ICC(3,1) = {icc_31:.3f})', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Mean of Measurements (px)', fontsize=11)
    ax2.set_ylabel('Difference (Algo - Expert) (px)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    
    # 6. 결과 출력 및 저장
    save_name = "Final_Reliability_Analysis.png"
    plt.savefig(save_name, dpi=400, bbox_inches='tight')
    plt.show()

    print("\n" + "="*50)
    print(f"📊 최종 분석 결과 (N={len(man_vals)})")
    print("-"*50)
    print(f"1. Pearson Correlation (r): {r_val:.4f} (p: {p_val:.4e})")
    print(f"2. Bland-Altman Bias: {bias:.4f}")
    print(f"3. Bland-Altman LoA: [{lower_loa:.4f}, {upper_loa:.4f}]")
    print(f"4. ICC(3,1): {icc_31:.4f}")
    print(f"5. ICC 95% Confidence Interval: {icc_ci}")
    print("="*50)
    print(f"✅ 그래프가 저장되었습니다: {os.path.abspath(save_name)}")

if __name__ == "__main__":
    main()