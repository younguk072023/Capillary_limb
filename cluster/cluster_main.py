import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data import load_and_preprocess_data
from clustering import perform_clustering, find_optimal_k
from stat_statistics import run_statistical_analysis
from visualization import run_dimension_reduction_plots, plot_stat_boxplot
from XAI_analysis import run_xai_analysis

def main():
    # 설정 
    csv_path = r"C:\Users\park_younguk\Desktop\analysis\limb_total.csv"
    result_dir = "analysis_results"
    feature_cols = ["arterial_diameter", "venous_diameter", "loop_diameter", "va_ratio", "avg_diameter3"]
    
    # 폰트 설정 (폰트를 가장 먼저 설정해야 모든 그래프에 적용됩니다)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    os.makedirs(result_dir, exist_ok=True)
    print("--- 분석 파이프라인 시작 ---")
    
    # 데이터 로드 및 전처리
    df = load_and_preprocess_data(csv_path)
    
    # 최적 K 분석 (df와 feature_cols가 준비된 후 실행)
    scaler_for_k = StandardScaler()
    X_scaled = scaler_for_k.fit_transform(df[feature_cols])
    find_optimal_k(X_scaled, max_k=7)
    
    # 본격적인 군집화 (수학적 근거로 확인된 K=3 적용)
    df, model, scaler = perform_clustering(df, feature_cols, k=3)
    
    # 통계 검정 및 결과 저장
    stat_df, _ = run_statistical_analysis(df, feature_cols)
    stat_df.to_csv(os.path.join(result_dir, "statistical_results.csv"), index=False)
    
    # 시각화 (PCA, t-SNE, UMAP)
    print("차원 축소 시각화 생성 중...")
    run_dimension_reduction_plots(df, scaler.transform(df[feature_cols]), result_dir)
    
    # 통계 박스플롯 생성
    print("통계 박스플롯 생성 중...")
    for feature in feature_cols:
        plot_stat_boxplot(df, feature, result_dir)
        
    # XAI 분석 (SHAP)
    print("SHAP 모델 해석 시작...")
    run_xai_analysis(df, feature_cols, result_dir)

    print(f"--- 분석 완료! 모든 결과는 '{result_dir}' 폴더에 저장되었습니다 ---")

if __name__ == "__main__":
    main()