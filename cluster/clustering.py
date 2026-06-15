from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

'''
1. 데이터 스케일링
2. KMeans 모델 학습 및 군집 할당
3. 최종 라벨이 포함된 df 반환
'''

def perform_clustering(df, feature_cols):
    
    X=df[feature_cols].copy()
    scaler = StandardScaler()   
    X_scaled = scaler.fit_transform(X)

    final_k=3
    model = KMeans(n_clusters=final_k, random_state=42, n_init=20)  #n_init은 초기화 반복 수행

    df["cluster"] = model.fit_predict(X_scaled)

    return df, model, scaler
