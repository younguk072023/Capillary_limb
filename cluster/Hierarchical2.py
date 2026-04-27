import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score




csv_path = "limb_total.csv"
image_dir = r"E:\MTL_dataset\iamge"

rep_copy_root = "cluster_representative_images"

out_csv = "limb_total_hierarchical_clustered.csv"
summary_csv = "limb_total_hierarchical_cluster_summary.csv"
score_csv = "limb_total_hierarchical_k_scores.csv"
rep_csv = "limb_total_hierarchical_representatives.csv"

df = pd.read_csv(csv_path, encoding = "utf-8-sig")
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

print("현재 컬럼명:")
print(df.columns.tolist())
print()

required_cols = ["filename","loop_length","arterial_diameter","venous_diameter"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

work_df = df[["filename","loop_length","arterial_diameter","venous_diameter"]].copy()

for c in ["loop_length","arterial_diameter"," venous_diameter"]:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

before_n = len(work_df)
work_df = work_df.dropna(subset=["loop_length", "arterial_diameter", "venous_diameter"]).reset_index(drop=True)
after_n = len(work_df)

print(f"전체 샘플 수: {before_n}")
print(f"결츨 제거 후 샘플 수:{after_n}")
print()

if len(work_df) < 3:
    raise ValueError("유효 샘플 수가 너무 적어서 clustering을 진행할 수 없음.")

eps = 1e-6

work_df["loop_diameter"] = work_df["loop_length"]

work_df["va_ratio"] = work_df["venous_diameter"]/(work_df["arterial_diameter"] + eps)

work_df["avg_diameter3"] = (
    work_df["arterial_diameter"]+
    work_df["venous_diameter"]+
    work_df["loop_diameter"]
)/3.0

#clustering에서 사용할 feature 선택
feature_cols = [
    "arterial_diameter",
    "venous_diameter",
    "loop_diameter",
    "va_ratio",
    "avg_diameter3"
]

x = work_df[feature_cols].copy()

print("최종 사용할 feature:")
print(feature_cols)
print()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

work_df["PC1"] = X_pca[:, 0]
work_df["PC2"] = X_pca[:, 1]

print("PCA 설명분산비:")
print(f"PC1:{pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2:{pca.explained_variance_ratio_[1]:.4f}")
print(f"합계:{(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])}")
print()



