import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

'''
1. 데이터 스케일링
2. KMeans 모델 학습 및 군집 할당
3. 최종 라벨이 포함된 df 반환
'''


def find_optimal_k(X_scaled, max_k=10):
   
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Silhouette Score Plot 
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111) # 서브플롯 추가
    ax1.plot(K_range, silhouette_scores, 'o-', color='#2e5a63', linewidth=2)
    ax1.set_xlabel("Number of clusters (K)", fontsize=12)
    ax1.set_ylabel("Silhouette score", fontsize=12)
    ax1.grid(alpha=0.3, linestyle='--')
    fig1.tight_layout()
    fig1.savefig("silhouette_score.png", dpi=300)
    plt.close(fig1)
    
    # Elbow Curve Plot 
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111) # 서브플롯 추가
    ax2.plot(K_range, inertias, 'o-', color='#958d88', linewidth=2)
    ax2.set_xlabel("Number of clusters (K)", fontsize=12)
    ax2.set_ylabel("Inertia", fontsize=12)
    ax2.grid(alpha=0.3, linestyle='--')
    fig2.tight_layout()
    fig2.savefig("elbow_curve.png", dpi=300)
    plt.close(fig2)
    
    print("그래프가 'silhouette_score.png'와 'elbow_curve.png'로 저장되었습니다.")

def perform_clustering(df, feature_cols, k=3):
    
    X=df[feature_cols].copy()
    scaler = StandardScaler()   
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)  #n_init은 초기화 반복 수행

    df["cluster"] = model.fit_predict(X_scaled)

    return df, model, scaler


