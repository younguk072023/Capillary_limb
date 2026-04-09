import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os


MANUAL_CSV = r"C:\Users\park_younguk\Desktop\analysis\total\image/manual_measurement_results.csv"
ALGO_CSV = r"C:\Users\park_younguk\Desktop\analysis\total\label\algo_gt_measurement_unified.csv"

def main():
    print("🚀 [Publication Style] 데이터 병합 및 신뢰도 분석 시작...")
    
    if not os.path.exists(MANUAL_CSV) or not os.path.exists(ALGO_CSV):
        print("CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 1. 데이터 로드
    df_man = pd.read_csv(MANUAL_CSV)
    df_alg = pd.read_csv(ALGO_CSV)

    # 2. 전처리 (확장자 제거 및 필터링)
    df_man['pure_name'] = df_man['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
    df_alg['pure_name'] = df_alg['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
    df_alg['Status'] = df_alg['Status'].str.strip()
    df_alg_success = df_alg[df_alg['Status'] == 'Success']

    df = pd.merge(df_man, df_alg_success, on="pure_name")
    
    if len(df) == 0:
        print("매칭되는 데이터가 없습니다!")
        return

    # 데이터 추출 (Arterial + Venous 통합)
    man_vals = np.concatenate([df['Manual_Arterial_Diameter(px)'].values, df['Manual_Venous_Diameter(px)'].values])
    alg_vals = np.concatenate([df['Algo_Arterial_Diameter(px)'].values, df['Algo_Venous_Diameter(px)'].values])

    # 통계 지표 계산
    r_val, p_val = pearsonr(man_vals, alg_vals)
    diff = alg_vals - man_vals
    avg = (alg_vals + man_vals) / 2
    bias = np.mean(diff)
    sd = np.std(diff)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    # 5. 시각화 설정 (Professional Style)
    sns.set_style("white") 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 캔버스 크기 설정 (좌우 대칭을 위해 가로 너비 조정)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 공통 마커 설정
    marker_style = dict(color='royalblue', alpha=0.5, s=35, edgecolors='white', linewidth=0.5)

    # --- Left: Pearson Correlation Plot ---
    ax1.scatter(man_vals, alg_vals, **marker_style, label='Measurements')
    
    # Identity Line (y=x)
    max_val = max(man_vals.max(), alg_vals.max()) * 1.05
    min_val = min(man_vals.min(), alg_vals.min()) * 0.95
    ax1.plot([min_val, max_val], [min_val, max_val], color='darkred', linestyle='--', linewidth=1.5, label='Identity Line (y=x)')
    
    # Regression Line
    m, b = np.polyfit(man_vals, alg_vals, 1)
    ax1.plot(man_vals, m*man_vals + b, color='black', alpha=0.8, linewidth=1, label=f'Regression (r={r_val:.3f})')

    ax1.set_title(f'Pearson Correlation (r={r_val:.3f})', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Professional Expert (px)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Proposed Algorithm (px)', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, loc='upper left', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)

    # --- Right: Bland-Altman Plot (여백 최적화 버전) ---
    # LoA Area Shading
    ax2.axhspan(lower_loa, upper_loa, color='royalblue', alpha=0.05)
    ax2.scatter(avg, diff, **marker_style)

    # Horizontal Lines (Bias and LoAs)
    ax2.axhline(bias, color='darkred', linestyle='-', linewidth=2)
    ax2.axhline(upper_loa, color='black', linestyle='--', linewidth=1.2)
    ax2.axhline(lower_loa, color='black', linestyle='--', linewidth=1.2)
    ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8)

   
    x_right_end = ax2.get_xlim()[1]
    x_text_pos = x_right_end - (x_right_end - ax2.get_xlim()[0]) * 0.02

    ax2.text(x_text_pos, bias + (sd*0.05), f'Mean: {bias:.2f}', va='bottom', ha='right', color='darkred', fontweight='bold', fontsize=10)
    ax2.text(x_text_pos, upper_loa + (sd*0.05), f'+1.96 SD: {upper_loa:.2f}', va='bottom', ha='right', color='black', fontsize=9)
    ax2.text(x_text_pos, lower_loa - (sd*0.05), f'-1.96 SD: {lower_loa:.2f}', va='top', ha='right', color='black', fontsize=9)

    ax2.set_title('Bland-Altman Agreement Plot', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Mean of Diameter (px)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Difference (Proposed - Expert) (px)', fontsize=12, fontweight='bold')
    
    # Y축 범위를 데이터에 맞게 최적화
    y_limit = max(abs(upper_loa), abs(lower_loa)) * 2.0
    ax2.set_ylim(bias - y_limit, bias + y_limit)
    ax2.grid(True, linestyle=':', alpha=0.3)

    # 테두리 강조
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    # 전체 레이아웃 자동 조정 (여백 최소화)
    plt.tight_layout()
    
    # 저장 및 출력
    save_path = "Reliability_Final_Publication.png"
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()

    print(f"분석 완료! 최종 매칭 샘플 수: {len(man_vals)}개")
    print(f"저장 위치: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()