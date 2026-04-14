import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ================= 설정 영역 =================
FILE_PATH = r"C:\Users\park_younguk\Desktop\analysis\effi_mask\final_mtl_algo_measurement.csv"
SAVE_PATH = r"C:\Users\park_younguk\Desktop\analysis\effi_mask\clustering_result.png"
N_CLUSTERS = 3  # 군집 개수 (보통 3~4개가 논문 설명에 좋습니다)
# =============================================

def perform_clustering():
    print("🚀 [학위 논문 데이터 분석] 군집화 및 패턴 분석 시작...")
    
    # 1. 데이터 로드 및 필터링
    df = pd.read_csv(FILE_PATH)
    clean_df = df[df['Status'] == 'Success'].copy()
    
    # 분석에 사용할 컬럼 (세동맥, 세정맥 직경)
    cols = ['Algo_Arterial_Diameter(px)', 'Algo_Venous_Diameter(px)']
    data = clean_df[cols].dropna()

    # 2. 통계적 이상치 제거 (Z-score 기준 3.0 이상 제거)
    # 학위 논문의 신뢰도를 높이기 위한 필수 과정입니다.
    z_scores = np.abs(stats.zscore(data))
    data = data[(z_scores < 3.0).all(axis=1)]
    print(f"✅ 이상치 제거 후 데이터 개수: {len(data)}개")

    # 3. 데이터 표준화 (Feature Scaling)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 4. K-means 군집화 실행
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(data_scaled)

    # 5. 결과 시각화 (Scatter Plot)
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # 색상 팔레트 설정
    palette = sns.color_palette("viridis", n_colors=N_CLUSTERS)
    
    scatter = sns.scatterplot(
        data=data, 
        x='Algo_Arterial_Diameter(px)', 
        y='Algo_Venous_Diameter(px)', 
        hue='Cluster', 
        palette=palette, 
        alpha=0.6, 
        edgecolor='w', 
        s=60
    )
    
    # 군집 중심(Centroid) 표시 (역변환 필요)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

    plt.title(f'Nailfold Capillary Diameter Clustering (K={N_CLUSTERS})', fontsize=15)
    plt.xlabel('Arterial Diameter (px)', fontsize=12)
    plt.ylabel('Venous Diameter (px)', fontsize=12)
    plt.legend()
    
    plt.savefig(SAVE_PATH)
    print(f"✅ 군집화 시각화 완료: {SAVE_PATH}")
    
    # 6. 군집별 특징 요약 통계량 출력
    print("\n📊 [군집별 평균 직경 분석]")
    summary = data.groupby('Cluster')[cols].mean()
    summary['Count'] = data.groupby('Cluster').size()
    print(summary)
    
    # 7. 분포 밀도 확인 (Jointplot)
    g = sns.jointplot(
        data=data, 
        x='Algo_Arterial_Diameter(px)', 
        y='Algo_Venous_Diameter(px)', 
        hue='Cluster', 
        palette=palette,
        kind='kde', 
        fill=True, 
        alpha=0.4
    )
    plt.show()

if __name__ == "__main__":
    perform_clustering()