"""
Hierarchical Clustering
"""

import os
os.environ["OMP_NUM_THREADS"] = "6"

import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.cluster.hierarchy import linkage, dendrogram


csv_path = "limb_total.csv"
image_dir = r"E:\MTL_dataset\image"

result_dir = "hierarchical_results"
os.makedirs(result_dir, exist_ok=True)

score_csv = os.path.join(result_dir, "limb_total_hierarchical_k_scores.csv")
pca_loading_csv = os.path.join(result_dir, "limb_total_hierarchical_pca_loadings.csv")

df = pd.read_csv(csv_path, encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

print("현재 컬럼명:")
print(df.columns.tolist())
print()

required_cols = [
    "filename",
    "loop_length",
    "arterial_diameter",
    "venous_diameter"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")



# 필요한 컬럼만 사용
work_df = df[
    [
        "filename",
        "loop_length",
        "arterial_diameter",
        "venous_diameter"
    ]
].copy()

for c in ["loop_length", "arterial_diameter", "venous_diameter"]:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

before_n = len(work_df)

work_df = work_df.dropna(
    subset=[
        "loop_length",
        "arterial_diameter",
        "venous_diameter"
    ]
).reset_index(drop=True)

after_n = len(work_df)

print(f"전체 샘플 수: {before_n}")
print(f"결측 제거 후 샘플 수: {after_n}")
print()

if len(work_df) < 3:
    raise ValueError("유효 샘플 수가 너무 적어서 clustering을 진행할 수 없음.")



# Feature Engineering
eps = 1e-6

# loop_length를 분석용 이름으로 통일
work_df["loop_diameter"] = work_df["loop_length"]

# 세정맥 / 세동맥 비율
work_df["va_ratio"] = (
    work_df["venous_diameter"] /
    (work_df["arterial_diameter"] + eps)
)

# 전체 직경 평균
work_df["avg_diameter3"] = (
    work_df["arterial_diameter"] +
    work_df["venous_diameter"] +
    work_df["loop_diameter"]
) / 3.0

feature_cols = [
    "arterial_diameter",
    "venous_diameter",
    "loop_diameter",
    "va_ratio",
    "avg_diameter3",
]

work_df = work_df.replace([np.inf, -np.inf], np.nan)
work_df = work_df.dropna(subset=feature_cols).reset_index(drop=True)

X = work_df[feature_cols].copy()

print("최종 사용할 feature:")
for i, col in enumerate(feature_cols, start=1):
    print(f"{i}. {col}")
print()

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

work_df["PC1"] = X_pca[:, 0]
work_df["PC2"] = X_pca[:, 1]

print("PCA 설명분산비:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(
    f"합계: "
    f"{(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.4f}"
)
print()

pca_loading = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=["PC1_loading", "PC2_loading"]
)

print("=== PCA Loading ===")
print(pca_loading.round(4))
print()

pca_loading.to_csv(pca_loading_csv, encoding="utf-8-sig")

max_k = min(7, len(work_df) - 1)
k_list = [k for k in [2, 3, 4, 5, 6, 7] if k <= max_k]

if len(k_list) == 0:
    raise ValueError("샘플 수가 너무 적어서 K 비교가 불가능함.")

score_rows = []

for k in k_list:
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    score_rows.append(
        {
            "K": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db
        }
    )

score_df = pd.DataFrame(score_rows)

print("=== Hierarchical Clustering K 비교 점수 ===")
print(score_df.to_string(index=False))
print()

score_df.to_csv(score_csv, index=False, encoding="utf-8-sig")


# silhouette 최고 K 선택
best_k = int(
    score_df
    .sort_values("silhouette", ascending=False)
    .iloc[0]["K"]
)

print(f"silhouette 기준 best K: {best_k}")
print()



# Hierarchical Clustering
final_k = best_k

print(f"최종 사용 K: {final_k}")
print()

hier_model = AgglomerativeClustering(
    n_clusters=final_k,
    linkage="ward"
)

# 원래 계층적 군집 번호 저장
work_df["hier_cluster_raw"] = hier_model.fit_predict(X_scaled)


# 0 = 세정맥 우세 비대칭형
# 1 = 전반적 확장형
# 2 = 소직경 균형형

if final_k == 3:
    raw_summary = (
        work_df
        .groupby("hier_cluster_raw")[feature_cols]
        .mean()
    )

    # 1) 세정맥 우세 비대칭형:
    # V/A ratio가 가장 큰 군집
    venous_dominant_raw = raw_summary["va_ratio"].idxmax()

    # 나머지 군집
    remaining_summary = raw_summary.drop(index=venous_dominant_raw)

    # 2) 전반적 확장형:
    # 평균 직경이 가장 큰 군집
    enlarged_raw = remaining_summary["avg_diameter3"].idxmax()

    # 3) 소직경 균형형:
    # 평균 직경이 가장 작은 군집
    small_balanced_raw = remaining_summary["avg_diameter3"].idxmin()

    label_map = {
        venous_dominant_raw: 0,
        enlarged_raw: 1,
        small_balanced_raw: 2,
    }

    work_df["hier_cluster"] = (
        work_df["hier_cluster_raw"]
        .map(label_map)
        .astype(int)
    )

    work_df["hier_cluster_type"] = work_df["hier_cluster"].map(
        {
            0: "venous_dominant_asymmetric",
            1: "global_enlargement",
            2: "small_balanced",
        }
    )

    print("=== Hierarchical raw label → KMeans-style label mapping ===")
    print(label_map)
    print()

else:
    # K=3이 아니면 자동 의미 매핑은 하지 않음
    # K=4 이상은 중간 균형형 등 추가 군집이 생기므로 별도 해석 필요
    work_df["hier_cluster"] = work_df["hier_cluster_raw"]
    work_df["hier_cluster_type"] = "not_mapped"

    print("final_k가 3이 아니므로 KMeans-style label mapping은 적용하지 않음.")
    print("K=4 이상은 별도 의미 해석 필요.")
    print()

rep_copy_root = os.path.join(
    result_dir,
    f"hierarchical_representative_images_K{final_k}"
)

out_csv = os.path.join(
    result_dir,
    f"limb_total_hierarchical_clustered_K{final_k}.csv"
)

summary_csv = os.path.join(
    result_dir,
    f"limb_total_hierarchical_cluster_summary_K{final_k}.csv"
)

std_csv = os.path.join(
    result_dir,
    f"limb_total_hierarchical_cluster_std_K{final_k}.csv"
)

rep_csv = os.path.join(
    result_dir,
    f"limb_total_hierarchical_representatives_K{final_k}.csv"
)
print(f"=== Hierarchical K={final_k} 군집별 샘플 수 ===")
cluster_counts = work_df["hier_cluster"].value_counts().sort_index()
print(cluster_counts)
print()

cluster_summary = (
    work_df
    .groupby("hier_cluster")[feature_cols]
    .mean()
    .round(3)
)

print(f"=== Hierarchical K={final_k} 군집별 평균 feature ===")
print(cluster_summary)
print()

cluster_std = (
    work_df
    .groupby("hier_cluster")[feature_cols]
    .std()
    .round(3)
)

print(f"=== Hierarchical K={final_k} 군집별 feature 표준편차 ===")
print(cluster_std)
print()

if final_k == 3:
    cluster_type_summary = (
        work_df
        .groupby(["hier_cluster", "hier_cluster_type"])
        .size()
        .reset_index(name="count")
    )

    print("=== Cluster 번호 의미 ===")
    print(cluster_type_summary.to_string(index=False))
    print()

work_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
cluster_summary.to_csv(summary_csv, encoding="utf-8-sig")
cluster_std.to_csv(std_csv, encoding="utf-8-sig")

print(f"샘플별 Hierarchical 결과 저장: {os.path.abspath(out_csv)}")
print(f"군집 요약표 저장: {os.path.abspath(summary_csv)}")
print(f"군집 표준편차 저장: {os.path.abspath(std_csv)}")
print(f"K 비교 점수 저장: {os.path.abspath(score_csv)}")
print()



# K별 평가 지표 그래프 저장
plt.figure(figsize=(8, 5))
plt.plot(score_df["K"], score_df["silhouette"], marker="o")
plt.axvline(final_k, linestyle=":", linewidth=1.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Hierarchical Clustering: Silhouette Score by K")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_silhouette_by_k.png"),
    dpi=300
)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(score_df["K"], score_df["calinski_harabasz"], marker="o")
plt.axvline(final_k, linestyle=":", linewidth=1.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Calinski-Harabasz score")
plt.title("Hierarchical Clustering: Calinski-Harabasz Score by K")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_calinski_by_k.png"),
    dpi=300
)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(score_df["K"], score_df["davies_bouldin"], marker="o")
plt.axvline(final_k, linestyle=":", linewidth=1.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Davies-Bouldin score")
plt.title("Hierarchical Clustering: Davies-Bouldin Score by K")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_davies_by_k.png"),
    dpi=300
)
plt.show()



# PCA 시각화
plt.figure(figsize=(8, 6))

for c in sorted(work_df["hier_cluster"].unique()):
    sub = work_df[work_df["hier_cluster"] == c]

    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        alpha=0.7,
        label=f"Cluster {c}",
        s=30
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
plt.title(f"PCA of Hierarchical Clustering Results (K={final_k})")
plt.legend()
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, f"hierarchical_pca_cluster_K{final_k}.png"),
    dpi=300
)
plt.show()

# Dendrogram

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(14, 7))
dendrogram(
    linked,
    no_labels=True,
    color_threshold=None
)
plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_dendrogram_full.png"),
    dpi=300
)
plt.show()

plt.figure(figsize=(12, 6))
dendrogram(
    linked,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True,
    color_threshold=None
)
plt.title("Truncated Hierarchical Dendrogram (last 30 merged clusters)")
plt.xlabel("Merged clusters")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_dendrogram_truncated.png"),
    dpi=300
)
plt.show()



# 15. 대표 이미지 경로 찾기 함수
def resolve_image_path(image_dir, filename):
    """
    CSV의 filename과 실제 이미지 파일 연결
    - image_dir 바로 아래에서 먼저 찾음
    - 확장자가 없으면 일반 이미지 확장자를 붙여서 찾음
    - 하위 폴더까지 재귀적으로 검색함
    """
    filename = str(filename).strip()

    exact_path = os.path.join(image_dir, filename)
    if os.path.exists(exact_path):
        return exact_path

    stem, ext = os.path.splitext(filename)

    if ext == "":
        for ext2 in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            cand = os.path.join(image_dir, stem + ext2)
            if os.path.exists(cand):
                return cand

    matches = glob.glob(os.path.join(image_dir, stem + ".*"))
    if len(matches) > 0:
        return matches[0]

    recursive_matches = glob.glob(
        os.path.join(image_dir, "**", filename),
        recursive=True
    )
    if len(recursive_matches) > 0:
        return recursive_matches[0]

    recursive_stem_matches = glob.glob(
        os.path.join(image_dir, "**", stem + ".*"),
        recursive=True
    )
    if len(recursive_stem_matches) > 0:
        return recursive_stem_matches[0]

    return None

rep_rows = []
missing_files = []

os.makedirs(rep_copy_root, exist_ok=True)

for c in sorted(work_df["hier_cluster"].unique()):
    cluster_idxs = work_df[work_df["hier_cluster"] == c].index.tolist()

    if len(cluster_idxs) == 0:
        continue

    # 해당 cluster의 표준화 feature 평균을 중심처럼 사용
    cluster_vectors = X_scaled[cluster_idxs]
    center = cluster_vectors.mean(axis=0)

    dist_list = []

    for idx in cluster_idxs:
        vec = X_scaled[idx]
        dist = np.linalg.norm(vec - center)
        dist_list.append((idx, dist))

    # 중심에 가까운 5개 선택
    dist_list = sorted(dist_list, key=lambda x: x[1])[:5]

    cluster_dir = os.path.join(rep_copy_root, f"hier_cluster_{c}")
    os.makedirs(cluster_dir, exist_ok=True)

    for idx, dist in dist_list:
        fname = work_df.loc[idx, "filename"]
        img_path = resolve_image_path(image_dir, fname)

        copied = "YES"
        reason = ""

        if img_path is None:
            copied = "NO"
            reason = "image_not_found"
            missing_files.append(fname)
        else:
            dst_path = os.path.join(cluster_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dst_path)

        row = {
            "hier_cluster": c,
            "filename": fname,
            "distance_to_cluster_mean": round(dist, 4),
            "arterial_diameter": round(work_df.loc[idx, "arterial_diameter"], 3),
            "venous_diameter": round(work_df.loc[idx, "venous_diameter"], 3),
            "loop_diameter": round(work_df.loc[idx, "loop_diameter"], 3),
            "va_ratio": round(work_df.loc[idx, "va_ratio"], 3),
            "avg_diameter3": round(work_df.loc[idx, "avg_diameter3"], 3),
            "copied": copied,
            "reason": reason,
        }

        if "hier_cluster_type" in work_df.columns:
            row["hier_cluster_type"] = work_df.loc[idx, "hier_cluster_type"]

        rep_rows.append(row)

rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(rep_csv, index=False, encoding="utf-8-sig")

print("=== Hierarchical 각 클러스터 대표 이미지 5개 ===")
print(rep_df.to_string(index=False))
print()

print(f"대표 샘플 목록 저장: {os.path.abspath(rep_csv)}")
print(f"대표 이미지 복사 폴더: {os.path.abspath(rep_copy_root)}")
print()

if len(missing_files) > 0:
    print(f"이미지를 못 찾은 파일 수: {len(missing_files)}")
    print("못 찾은 파일 예시 10개:")
    print(missing_files[:10])
    print()
else:
    print("모든 대표 이미지 복사 성공")
    print()

print("=========================================")
print("Hierarchical Clustering 분석 완료")
print("K-Means와 동일한 의미 체계로 label 정렬 완료")
print("Cluster 0 = 세정맥 우세 비대칭형")
print("Cluster 1 = 전반적 확장형")
print("Cluster 2 = 소직경 균형형")
print("=========================================")