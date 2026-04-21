import os
os.environ["OMP_NUM_THREADS"] = "6"   # Windows + MKL warning 완화용, sklearn import 전에 두기

import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# =========================================
# 0. 경로 설정
# =========================================
csv_path = "limb_total.csv"
image_dir = r"E:\MTL_dataset\image"

# 대표 이미지 복사 저장 폴더
rep_copy_root = "cluster_representative_images"

# 결과 CSV 저장명
out_csv = "limb_total_clustered.csv"
summary_csv = "limb_total_cluster_summary.csv"
score_csv = "limb_total_k_scores.csv"
rep_csv = "limb_total_cluster_representatives.csv"

# CSV 불러오기
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 혹시 모를 공백/BOM 제거
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

print("현재 컬럼명:")
print(df.columns.tolist())
print()


# =========================================
# 2. 필수 컬럼 확인
# =========================================
required_cols = ["filename", "loop_length", "arterial_diameter", "venous_diameter"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

# 필요한 컬럼만 사용
work_df = df[["filename", "loop_length", "arterial_diameter", "venous_diameter"]].copy()

# 숫자형 변환
for c in ["loop_length", "arterial_diameter", "venous_diameter"]:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

# 결측 제거
before_n = len(work_df)
work_df = work_df.dropna(subset=["loop_length", "arterial_diameter", "venous_diameter"]).reset_index(drop=True)
after_n = len(work_df)

print(f"전체 샘플 수: {before_n}")
print(f"결측 제거 후 샘플 수: {after_n}")
print()

if len(work_df) < 3:
    raise ValueError("유효 샘플 수가 너무 적어서 clustering을 진행할 수 없음.")


# =========================================
# 3. Feature Engineering
# =========================================
eps = 1e-6   # 0으로 나누기 방지

# loop_length를 분석용 이름으로 통일
work_df["loop_diameter"] = work_df["loop_length"]

# 세정맥 / 세동맥 비율
work_df["va_ratio"] = work_df["venous_diameter"] / (work_df["arterial_diameter"] + eps)

# 전체 capillary 크기 요약값
work_df["avg_diameter3"] = (
    work_df["arterial_diameter"] +
    work_df["venous_diameter"] +
    work_df["loop_diameter"]
) / 3.0

# Clustering에 사용할 feature 선택
feature_cols = [
    "arterial_diameter",
    "venous_diameter",
    "loop_diameter",
    "va_ratio",
    "avg_diameter3",
]

X = work_df[feature_cols].copy()

print("최종 사용할 feature:")
print(feature_cols)
print()


# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA 2차원 축소
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

work_df["PC1"] = X_pca[:, 0]
work_df["PC2"] = X_pca[:, 1]

print("PCA 설명분산비:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"합계: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.4f}")
print()


# =========================================
# 7. K 비교 (2~7)
# =========================================
max_k = min(7, len(work_df) - 1)
k_list = [k for k in [2, 3, 4, 5, 6, 7] if k <= max_k]

if len(k_list) == 0:
    raise ValueError("샘플 수가 너무 적어서 KMeans 비교가 불가능함.")

score_rows = []

for k in k_list:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    score_rows.append({
        "K": k,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db
    })

score_df = pd.DataFrame(score_rows)

print("=== K 비교 점수 ===")
print(score_df.to_string(index=False))
print()

# silhouette 최고인 K 선택
best_k = int(score_df.sort_values("silhouette", ascending=False).iloc[0]["K"])
print(f"자동 선택된 K (silhouette 기준): {best_k}")
print()


# =========================================
# 8. 최종 KMeans
# =========================================
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
work_df["cluster"] = final_model.fit_predict(X_scaled)


# =========================================
# 9. 군집별 샘플 수
# =========================================
print("=== 군집별 샘플 수 ===")
print(work_df["cluster"].value_counts().sort_index())
print()


# =========================================
# 10. 군집별 평균 feature
# =========================================
cluster_summary = work_df.groupby("cluster")[feature_cols].mean().round(3)

print("=== 군집별 평균 feature ===")
print(cluster_summary)
print()


# =========================================
# 11. 결과 저장
# =========================================
work_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
cluster_summary.to_csv(summary_csv, encoding="utf-8-sig")
score_df.to_csv(score_csv, index=False, encoding="utf-8-sig")

print(f"샘플별 군집 결과 저장: {os.path.abspath(out_csv)}")
print(f"군집 요약표 저장: {os.path.abspath(summary_csv)}")
print(f"K 비교 점수 저장: {os.path.abspath(score_csv)}")
print()


# =========================================
# 12. PCA 시각화
# =========================================
plt.figure(figsize=(8, 6))

for c in sorted(work_df["cluster"].unique()):
    sub = work_df[work_df["cluster"] == c]
    plt.scatter(sub["PC1"], sub["PC2"], alpha=0.7, label=f"Cluster {c}", s=30)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
plt.title(f"PCA of Capillary Features (K={best_k})")
plt.legend()
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.show()


# =========================================
# 13. filename으로 실제 이미지 경로 찾는 함수
# =========================================
def resolve_image_path(image_dir, filename):
    """
    CSV의 filename과 실제 이미지 파일 연결
    - filename에 확장자가 있으면 그대로 찾음
    - 없으면 일반적인 이미지 확장자를 붙여서 찾음
    - 그래도 없으면 stem이 같은 파일을 glob으로 찾음
    """
    filename = str(filename).strip()

    # 1) 그대로 찾기
    exact_path = os.path.join(image_dir, filename)
    if os.path.exists(exact_path):
        return exact_path

    # 2) 확장자가 없는 경우 확장자 붙여서 찾기
    stem, ext = os.path.splitext(filename)
    if ext == "":
        for ext2 in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            cand = os.path.join(image_dir, stem + ext2)
            if os.path.exists(cand):
                return cand

    # 3) 같은 stem을 가진 아무 파일이나 찾기
    matches = glob.glob(os.path.join(image_dir, stem + ".*"))
    if len(matches) > 0:
        return matches[0]

    return None


# =========================================
# 14. 각 클러스터 대표 샘플 5개 추출
#     (군집 중심에 가장 가까운 샘플)
# =========================================
centers = final_model.cluster_centers_
rep_rows = []
missing_files = []

os.makedirs(rep_copy_root, exist_ok=True)

for c in range(best_k):
    cluster_df = work_df[work_df["cluster"] == c].copy()
    idxs = cluster_df.index.tolist()

    if len(idxs) == 0:
        continue

    center = centers[c]
    dist_list = []

    for idx in idxs:
        vec = X_scaled[idx]
        dist = np.linalg.norm(vec - center)
        dist_list.append((idx, dist))

    # 중심에 가장 가까운 5개 선택
    dist_list = sorted(dist_list, key=lambda x: x[1])[:5]

    # cluster별 저장 폴더 생성
    cluster_dir = os.path.join(rep_copy_root, f"cluster_{c}")
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

        rep_rows.append({
            "cluster": c,
            "filename": fname,
            "distance_to_center": round(dist, 4),
            "arterial_diameter": round(work_df.loc[idx, "arterial_diameter"], 3),
            "venous_diameter": round(work_df.loc[idx, "venous_diameter"], 3),
            "loop_diameter": round(work_df.loc[idx, "loop_diameter"], 3),
            "va_ratio": round(work_df.loc[idx, "va_ratio"], 3),
            "avg_diameter3": round(work_df.loc[idx, "avg_diameter3"], 3),
            "copied": copied,
            "reason": reason,
        })

rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(rep_csv, index=False, encoding="utf-8-sig")

print("=== 각 클러스터 대표 이미지 5개 ===")
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