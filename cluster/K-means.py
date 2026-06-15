"""
K-Means Clustering + Kruskal-Wallis + Effect Size + Dunn Post-hoc + Boxplot
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from scipy.stats import kruskal

# Dunn post-hoc test
try:
    import scikit_posthocs as sp
    SCIPOSTHOCS_AVAILABLE = True
except ImportError:
    SCIPOSTHOCS_AVAILABLE = False
    print("scikit-posthocs가 설치되어 있지 않음.")
    print("Dunn post-hoc test를 수행하려면 아래 명령어로 설치:")
    print("pip install scikit-posthocs")
    print()



# 기본 경로 설정
csv_path = r"C:\Users\park_younguk\Desktop\analysis\limb_total.csv" # 각 직경이 정리되어있는 csv 파일
image_dir = r"D:\usb\MTL_dataset\image" # 이미지 정리되어있는 파일

result_dir = "kmeans_results"
os.makedirs(result_dir, exist_ok=True)

rep_copy_root = os.path.join(result_dir, "kmeans_representative_images")
out_csv = os.path.join(result_dir, "limb_total_kmeans_clustered.csv")
summary_csv = os.path.join(result_dir, "limb_total_kmeans_cluster_summary.csv")
std_csv = os.path.join(result_dir, "limb_total_kmeans_cluster_std.csv")
score_csv = os.path.join(result_dir, "limb_total_kmeans_k_scores.csv")
rep_csv = os.path.join(result_dir, "limb_total_kmeans_representatives.csv")
pca_loading_csv = os.path.join(result_dir, "limb_total_kmeans_pca_loadings.csv")
kruskal_csv = os.path.join(result_dir, "limb_total_kmeans_kruskal.csv")
dunn_summary_csv = os.path.join(result_dir, "limb_total_kmeans_dunn_summary.csv")
dunn_dir = os.path.join(result_dir, "dunn_posthoc")
boxplot_dir = os.path.join(result_dir, "cluster_boxplots")

os.makedirs(dunn_dir, exist_ok=True)
os.makedirs(boxplot_dir, exist_ok=True)


# CSV 로드 및 컬럼 확인
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

work_df["loop_diameter"] = work_df["loop_length"]

work_df["va_ratio"] = (
    work_df["venous_diameter"] /
    (work_df["arterial_diameter"] + eps)
)

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



# PCA

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

# K-Means K 비교
max_k = min(7, len(work_df) - 1)
k_list = [k for k in [2, 3, 4, 5, 6, 7] if k <= max_k]

if len(k_list) == 0:
    raise ValueError("샘플 수가 너무 적어서 KMeans 비교가 불가능함.")

score_rows = []

for k in k_list:    #첫 번쨰 K-means는 어떤 K가 좋은지 비교하기 위한 임시 모델
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20   #초기 중심점으로 20번 반복 실행한 뒤, 그중 가장 좋은 결과 선택
    )

    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    score_rows.append(
        {
            "K": k,
            "inertia": inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db
        }
    )

score_df = pd.DataFrame(score_rows)

print("=== K-Means K 비교 점수 ===")
print(score_df.to_string(index=False))
print()

score_df.to_csv(score_csv, index=False, encoding="utf-8-sig")


best_k = int(
    score_df
    .sort_values("silhouette", ascending=False)
    .iloc[0]["K"]
)

print(f"silhouette 기준 best K: {best_k}")
print()


# 최종 K-Means
final_k = best_k

print(f"최종 사용 K: {final_k}")
print()

final_model = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=20
)

work_df["cluster_raw"] = final_model.fit_predict(X_scaled)

if final_k == 3:
    raw_summary = (
        work_df
        .groupby("cluster_raw")[feature_cols]
        .mean()
    )

    # 1) 세정맥 우세 비대칭형:
    # V/A ratio가 가장 큰 군집
    venous_dominant_raw = raw_summary["va_ratio"].idxmax()

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

    work_df["cluster"] = (
        work_df["cluster_raw"]
        .map(label_map)
        .astype(int)
    )

    work_df["cluster_type"] = work_df["cluster"].map(
        {
            0: "Asymmetric",
            1: "Dilated",
            2: "Blanced",
        }
    )

    print("=== K-Means raw label → 의미 기반 label mapping ===")
    print(label_map)
    print()

else:
    work_df["cluster"] = work_df["cluster_raw"]
    work_df["cluster_type"] = "not_mapped"

    print("final_k가 3이 아니므로 의미 기반 label mapping은 적용하지 않음.")
    print("K=4 이상은 별도 의미 해석 필요.")
    print()

# 군집별 요약 통계
print(f"=== K-Means K={final_k} 군집별 샘플 수 ===")
cluster_counts = work_df["cluster"].value_counts().sort_index()
print(cluster_counts)
print()

cluster_summary = (
    work_df
    .groupby("cluster")[feature_cols]
    .mean()
    .round(3)
)

print(f"=== K-Means K={final_k} 군집별 평균 feature ===")
print(cluster_summary)
print()

cluster_std = (
    work_df
    .groupby("cluster")[feature_cols]
    .std()
    .round(3)
)

print(f"=== K-Means K={final_k} 군집별 feature 표준편차 ===")
print(cluster_std)
print()

if final_k == 3:
    cluster_type_summary = (
        work_df
        .groupby(["cluster", "cluster_type"])
        .size()
        .reset_index(name="count")
    )

    print("=== Cluster 번호 의미 ===")
    print(cluster_type_summary.to_string(index=False))
    print()

# Kruskal-Wallis Test
cluster_col = "cluster"

kruskal_rows = []

print("=== Kruskal-Wallis test: K-Means 군집 간 feature 차이 검정 ===")

for feature in feature_cols:
    groups = []
    group_sizes = {}

    for c in sorted(work_df[cluster_col].unique()):
        values = work_df.loc[work_df[cluster_col] == c, feature].dropna()
        groups.append(values)
        group_sizes[f"cluster_{c}_n"] = len(values)

    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        print(f"{feature}: 검정 불가 - 군집 수 또는 샘플 수 부족")
        continue

    h_stat, p_value = kruskal(*groups)

    row = {
        "feature": feature,
        "H_statistic": h_stat,
        "p_value": p_value,
        "significant_p_0.05": p_value < 0.05
    }

    row.update(group_sizes)
    kruskal_rows.append(row)

    print(
        f"{feature}: "
        f"H = {h_stat:.4f}, "
        f"p = {p_value:.6f}, "
        f"significant = {p_value < 0.05}"
    )

kruskal_df = pd.DataFrame(kruskal_rows)

if len(kruskal_df) > 0:
    kruskal_df["p_bonferroni"] = (
        kruskal_df["p_value"] * len(kruskal_df)
    ).clip(upper=1.0)

    kruskal_df["significant_bonferroni_0.05"] = (
        kruskal_df["p_bonferroni"] < 0.05
    )

    # Effect size: epsilon squared
    n_total = len(work_df)
    k_groups = work_df["cluster"].nunique()

    if n_total > k_groups:
        kruskal_df["epsilon_squared"] = (
            (kruskal_df["H_statistic"] - k_groups + 1) /
            (n_total - k_groups)
        )
        kruskal_df["epsilon_squared"] = (
            kruskal_df["epsilon_squared"]
            .clip(lower=0, upper=1)
        )
    else:
        kruskal_df["epsilon_squared"] = np.nan

    kruskal_df = kruskal_df.sort_values("p_value")

    print()
    print("=== Kruskal-Wallis 결과 요약 + effect size ===")
    print(kruskal_df.to_string(index=False))
    print()

    kruskal_df.to_csv(
        kruskal_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Kruskal-Wallis 결과 저장: {os.path.abspath(kruskal_csv)}")
    print()

else:
    print("Kruskal-Wallis 결과가 생성되지 않았음.")
    print()



# Dunn Post-hoc Test
dunn_summary_rows = []

if len(kruskal_df) > 0 and SCIPOSTHOCS_AVAILABLE:
    print("=== Dunn post-hoc test: K-Means 군집 간 쌍별 비교 ===")

    for feature in feature_cols:
        matched = kruskal_df.loc[
            kruskal_df["feature"] == feature,
            "p_bonferroni"
        ]

        if len(matched) == 0:
            print(f"{feature}: Kruskal-Wallis 결과 없음. Dunn test 생략")
            continue

        p_val = matched.values[0]

        if p_val >= 0.05:
            print(f"{feature}: Bonferroni 보정 후 유의하지 않아 Dunn test 생략")
            continue

        dunn_result = sp.posthoc_dunn(
            work_df,
            val_col=feature,
            group_col="cluster",
            p_adjust="bonferroni"
        )

        dunn_csv = os.path.join(
            dunn_dir,
            f"dunn_posthoc_{feature}_K{final_k}.csv"
        )

        dunn_result.to_csv(dunn_csv, encoding="utf-8-sig")

        print()
        print(f"=== Dunn post-hoc: {feature} ===")
        print(dunn_result)
        print(f"저장: {os.path.abspath(dunn_csv)}")

        clusters = sorted(work_df["cluster"].unique())

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1 = clusters[i]
                c2 = clusters[j]

                try:
                    pair_p = dunn_result.loc[c1, c2]
                except KeyError:
                    pair_p = dunn_result.loc[str(c1), str(c2)]

                dunn_summary_rows.append(
                    {
                        "feature": feature,
                        "comparison": f"cluster_{c1}_vs_cluster_{c2}",
                        "p_adjusted": pair_p,
                        "significant_0.05": pair_p < 0.05
                    }
                )

    dunn_summary_df = pd.DataFrame(dunn_summary_rows)

    if len(dunn_summary_df) > 0:
        dunn_summary_df.to_csv(
            dunn_summary_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print("=== Dunn post-hoc summary ===")
        print(dunn_summary_df.to_string(index=False))
        print(f"Dunn summary 저장: {os.path.abspath(dunn_summary_csv)}")
        print()
    else:
        print("Dunn post-hoc summary가 생성되지 않았음.")
        print()

elif not SCIPOSTHOCS_AVAILABLE:
    print("scikit-posthocs가 없어 Dunn post-hoc test를 건너뜀.")
    print("필요하면 터미널에서 다음 명령 실행:")
    print("pip install scikit-posthocs")
    print()
else:
    print("Kruskal-Wallis 결과가 없어 Dunn post-hoc test를 건너뜀.")
    print()



# Cluster-wise Boxplot
print("=== Cluster-wise boxplot 저장 ===")

for feature in feature_cols:
    plt.figure(figsize=(7, 5))

    sorted_clusters = sorted(work_df["cluster"].unique())

    data_to_plot = [
        work_df.loc[work_df["cluster"] == c, feature].dropna()
        for c in sorted_clusters
    ]

    label_names = []

    for c in sorted_clusters:
        if final_k == 3:
            c_type = (
                work_df.loc[work_df["cluster"] == c, "cluster_type"]
                .iloc[0]
            )
            label_names.append(f"C{c}\n{c_type}")
        else:
            label_names.append(f"Cluster {c}")

    plt.boxplot(
        data_to_plot,
        labels=label_names,
        showfliers=False
    )

    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.title(f"K-Means: {feature} by cluster")
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()

    save_path = os.path.join(
        boxplot_dir,
        f"boxplot_{feature}_K{final_k}.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Boxplot 저장: {os.path.abspath(save_path)}")

print()



# 군집 해석용 통합 요약표 저장
cluster_summary_mean = (
    work_df
    .groupby("cluster")[feature_cols]
    .mean()
)

cluster_summary_std = (
    work_df
    .groupby("cluster")[feature_cols]
    .std()
)

cluster_summary_count = (
    work_df
    .groupby("cluster")
    .size()
    .rename("n")
)

cluster_interpretation = cluster_summary_mean.copy()
cluster_interpretation.columns = [
    f"{c}_mean" for c in cluster_interpretation.columns
]

for col in feature_cols:
    cluster_interpretation[f"{col}_std"] = cluster_summary_std[col]

cluster_interpretation["n"] = cluster_summary_count

if "cluster_type" in work_df.columns:
    cluster_type_map = (
        work_df
        .groupby("cluster")["cluster_type"]
        .first()
    )
    cluster_interpretation["cluster_type"] = cluster_type_map

cluster_interpretation = cluster_interpretation.reset_index()

cluster_interpretation_csv = os.path.join(
    result_dir,
    f"limb_total_kmeans_cluster_interpretation_K{final_k}.csv"
)

cluster_interpretation.to_csv(
    cluster_interpretation_csv,
    index=False,
    encoding="utf-8-sig"
)

print("=== K-Means 군집 해석용 통합 요약표 ===")
print(cluster_interpretation.round(3).to_string(index=False))
print(f"군집 해석용 통합 요약표 저장: {os.path.abspath(cluster_interpretation_csv)}")
print()



# 결과 CSV 저장
work_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
cluster_summary.to_csv(summary_csv, encoding="utf-8-sig")
cluster_std.to_csv(std_csv, encoding="utf-8-sig")

print(f"샘플별 K-Means 결과 저장: {os.path.abspath(out_csv)}")
print(f"군집 요약표 저장: {os.path.abspath(summary_csv)}")
print(f"군집 표준편차 저장: {os.path.abspath(std_csv)}")
print(f"K 비교 점수 저장: {os.path.abspath(score_csv)}")
print()

# K별 평가 지표 그래프 저장

# Elbow plot: inertia by K
plt.figure(figsize=(8, 5))
plt.plot(score_df["K"], score_df["inertia"], marker="o")
plt.axvline(final_k, linestyle=":", linewidth=1.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means: Elbow Curve")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "kmeans_elbow_inertia_by_k.png"),
    dpi=300
)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(score_df["K"], score_df["silhouette"], marker="o")
plt.axvline(final_k, linestyle=":", linewidth=1.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette score")
plt.title("K-Means: Silhouette Score by K")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "kmeans_silhouette_by_k.png"),
    dpi=300
)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(score_df["K"], score_df["calinski_harabasz"], marker="o")
plt.axvline(final_k, linestyle=":", linewidth=1.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Calinski-Harabasz score")
plt.title("K-Means: Calinski-Harabasz Score by K")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "kmeans_calinski_by_k.png"),
    dpi=300
)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(score_df["K"], score_df["davies_bouldin"], marker="o")
plt.axvline(final_k, linestyle=":", linewidth=1.5)
plt.xlabel("Number of clusters (K)")
plt.ylabel("Davies-Bouldin score")
plt.title("K-Means: Davies-Bouldin Score by K")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "kmeans_davies_by_k.png"),
    dpi=300
)
plt.show()



# PCA 시각화
plt.figure(figsize=(8, 6))

for c in sorted(work_df["cluster"].unique()):
    sub = work_df[work_df["cluster"] == c]

    legend_label = f"Cluster {c}"

    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        alpha=0.7,
        label=legend_label,
        s=30
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
#plt.title(f"PCA of K-Means Clustering Results (K={final_k})")
plt.legend()
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, f"kmeans_pca_cluster_K{final_k}.png"),
    dpi=300
)
plt.show()



# filename으로 실제 이미지 경로 찾기 함수
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



# 대표 이미지 선택 및 복사
rep_rows = []
missing_files = []

os.makedirs(rep_copy_root, exist_ok=True)

for c in sorted(work_df["cluster"].unique()):
    cluster_idxs = work_df[work_df["cluster"] == c].index.tolist()

    if len(cluster_idxs) == 0:
        continue

    cluster_vectors = X_scaled[cluster_idxs]
    center = cluster_vectors.mean(axis=0)

    dist_list = []

    for idx in cluster_idxs:
        vec = X_scaled[idx]
        dist = np.linalg.norm(vec - center)
        dist_list.append((idx, dist))

    dist_list = sorted(dist_list, key=lambda x: x[1])[:5]

    cluster_dir = os.path.join(rep_copy_root, f"kmeans_cluster_{c}")
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
            "cluster": c,
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

        if "cluster_type" in work_df.columns:
            row["cluster_type"] = work_df.loc[idx, "cluster_type"]

        rep_rows.append(row)

rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(rep_csv, index=False, encoding="utf-8-sig")

print("=== K-Means 각 클러스터 대표 이미지 5개 ===")
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



# 최종 출력
print("=========================================")
print("K-Means Clustering 분석 완료")

if final_k == 3:
    print("Cluster 0 = 세정맥 우세 비대칭형")
    print("Cluster 1 = 전반적 확장형")
    print("Cluster 2 = 소직경 균형형")
else:
    print("final_k가 3이 아니므로 cluster 의미는 별도 해석 필요")

print("Kruskal-Wallis 검정 완료")
print("Effect size 계산 완료")

if SCIPOSTHOCS_AVAILABLE:
    print("Dunn post-hoc test 완료")
else:
    print("Dunn post-hoc test는 scikit-posthocs 미설치로 건너뜀")

print("Cluster-wise boxplot 저장 완료")
print("=========================================")

# =====================================================================
# 3차원 산점도 시각화 (X: Arterial, Y: Venous, Z: Loop)
# =====================================================================
print("=== 3D Scatter plot 저장 ===")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for c in sorted(work_df["cluster"].unique()):
    sub = work_df[work_df["cluster"] == c]

    # 라벨링 설정
    if final_k == 3 and "cluster_type" in work_df.columns:
        label_name = sub["cluster_type"].iloc[0]
        legend_label = f"Cluster {c}: {label_name}"
    else:
        legend_label = f"Cluster {c}"

    # 3D 산점도 그리기
    ax.scatter(
        sub["arterial_diameter"],
        sub["venous_diameter"],
        sub["loop_diameter"],
        alpha=0.7,
        s=40,
        label=legend_label,
        edgecolors='w',       # 점의 테두리를 하얗게 주어 겹칠 때 구분이 잘 되게 함
        linewidth=0.5
    )

# 축 이름 설정
ax.set_xlabel("Arterial Diameter (px)", labelpad=10)
ax.set_ylabel("Venous Diameter (px)", labelpad=10)
ax.set_zlabel("Loop Diameter (px)", labelpad=10)
# ax.set_title(f"3D Scatter Plot of Capillary Clusters (K={final_k})") # 필요 시 주석 해제

# 시야각(Viewing angle) 조절 
# elev: 위아래 각도 (기본 30도)
# azim: 좌우 회전 각도 (기본 -60도)
# 그래프가 저장된 후, 군집이 가장 잘 나뉘어 보이는 각도를 찾아 이 숫자를 변경하시면 됩니다.
ax.view_init(elev=20, azim=45) 

plt.legend()
plt.tight_layout()

save_path_3d = os.path.join(result_dir, f"kmeans_3d_scatter_K{final_k}.png")
plt.savefig(save_path_3d, dpi=400)
plt.show()

print(f"3D Scatter plot 저장 완료: {os.path.abspath(save_path_3d)}")
print()

"""
K-Means Clustering + Kruskal-Wallis + Dunn Post-hoc + t-SNE/UMAP + RF & SHAP(XAI)
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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from scipy.stats import kruskal

# 1. 추가 패키지 확인 (Dunn, UMAP, SHAP)
try:
    import scikit_posthocs as sp
    SCIPOSTHOCS_AVAILABLE = True
except ImportError:
    SCIPOSTHOCS_AVAILABLE = False
    print("scikit-posthocs 미설치. (pip install scikit-posthocs)")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("umap-learn 미설치. UMAP 시각화를 건너뜁니다. (pip install umap-learn)")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("shap 미설치. XAI 분석을 건너뜁니다. (pip install shap)")

print()


# 2. 기본 경로 설정
csv_path = r"C:\Users\park_younguk\Desktop\analysis\limb_total.csv"
image_dir = r"D:\usb\MTL_dataset\image"

result_dir = "kmeans_advanced_results"
os.makedirs(result_dir, exist_ok=True)

rep_copy_root = os.path.join(result_dir, "kmeans_representative_images")
out_csv = os.path.join(result_dir, "limb_total_kmeans_clustered.csv")
summary_csv = os.path.join(result_dir, "limb_total_kmeans_cluster_summary.csv")
std_csv = os.path.join(result_dir, "limb_total_kmeans_cluster_std.csv")
score_csv = os.path.join(result_dir, "limb_total_kmeans_k_scores.csv")
rep_csv = os.path.join(result_dir, "limb_total_kmeans_representatives.csv")
pca_loading_csv = os.path.join(result_dir, "limb_total_kmeans_pca_loadings.csv")
rf_importance_csv = os.path.join(result_dir, "limb_total_rf_feature_importance.csv")

kruskal_csv = os.path.join(result_dir, "limb_total_kmeans_kruskal.csv")
dunn_summary_csv = os.path.join(result_dir, "limb_total_kmeans_dunn_summary.csv")
dunn_dir = os.path.join(result_dir, "dunn_posthoc")
boxplot_dir = os.path.join(result_dir, "cluster_boxplots")
xai_dir = os.path.join(result_dir, "xai_plots")

os.makedirs(dunn_dir, exist_ok=True)
os.makedirs(boxplot_dir, exist_ok=True)
os.makedirs(xai_dir, exist_ok=True)


# 3. CSV 로드 및 데이터 전처리
df = pd.read_csv(csv_path, encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

required_cols = ["filename", "loop_length", "arterial_diameter", "venous_diameter"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

work_df = df[required_cols].copy()
for c in ["loop_length", "arterial_diameter", "venous_diameter"]:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

work_df = work_df.dropna(subset=required_cols).reset_index(drop=True)

if len(work_df) < 3:
    raise ValueError("유효 샘플 수가 부족합니다.")

# Feature Engineering
eps = 1e-6
work_df["loop_diameter"] = work_df["loop_length"]
work_df["va_ratio"] = work_df["venous_diameter"] / (work_df["arterial_diameter"] + eps)
work_df["avg_diameter3"] = (work_df["arterial_diameter"] + work_df["venous_diameter"] + work_df["loop_diameter"]) / 3.0

feature_cols = ["arterial_diameter", "venous_diameter", "loop_diameter", "va_ratio", "avg_diameter3"]

work_df = work_df.replace([np.inf, -np.inf], np.nan)
work_df = work_df.dropna(subset=feature_cols).reset_index(drop=True)

X = work_df[feature_cols].copy()

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. 차원 축소 (PCA, t-SNE, UMAP)
print("=== 차원 축소 (PCA, t-SNE, UMAP) 진행 중 ===")
# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
work_df["PC1"] = X_pca[:, 0]
work_df["PC2"] = X_pca[:, 1]
pca_loading = pd.DataFrame(pca.components_.T, index=feature_cols, columns=["PC1_loading", "PC2_loading"])
pca_loading.to_csv(pca_loading_csv, encoding="utf-8-sig")

# t-SNE
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)
work_df["tSNE1"] = X_tsne[:, 0]
work_df["tSNE2"] = X_tsne[:, 1]

# UMAP
if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    work_df["UMAP1"] = X_umap[:, 0]
    work_df["UMAP2"] = X_umap[:, 1]
print("차원 축소 완료.\n")


# 5. K-Means K 비교 및 최적 K 선정
max_k = min(7, len(work_df) - 1)
k_list = [k for k in [2, 3, 4, 5, 6, 7] if k <= max_k]
score_rows = []

for k in k_list:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    score_rows.append({
        "K": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels)
    })

score_df = pd.DataFrame(score_rows)
score_df.to_csv(score_csv, index=False, encoding="utf-8-sig")

best_k = int(score_df.sort_values("silhouette", ascending=False).iloc[0]["K"])
final_k = best_k

final_model = KMeans(n_clusters=final_k, random_state=42, n_init=20)
work_df["cluster_raw"] = final_model.fit_predict(X_scaled)

if final_k == 3:
    raw_summary = work_df.groupby("cluster_raw")[feature_cols].mean()
    venous_dominant_raw = raw_summary["va_ratio"].idxmax()
    remaining_summary = raw_summary.drop(index=venous_dominant_raw)
    enlarged_raw = remaining_summary["avg_diameter3"].idxmax()
    small_balanced_raw = remaining_summary["avg_diameter3"].idxmin()

    label_map = {venous_dominant_raw: 0, enlarged_raw: 1, small_balanced_raw: 2}
    work_df["cluster"] = work_df["cluster_raw"].map(label_map).astype(int)
    work_df["cluster_type"] = work_df["cluster"].map({0: "Asymmetric", 1: "Dilated", 2: "Balanced"})
else:
    work_df["cluster"] = work_df["cluster_raw"]
    work_df["cluster_type"] = "not_mapped"

cluster_summary = work_df.groupby("cluster")[feature_cols].mean().round(3)
cluster_std = work_df.groupby("cluster")[feature_cols].std().round(3)


# 6. 통계 분석 (Kruskal-Wallis & Dunn)
kruskal_rows = []
for feature in feature_cols:
    groups = [work_df.loc[work_df["cluster"] == c, feature].dropna() for c in sorted(work_df["cluster"].unique())]
    if len(groups) < 2: continue
    h_stat, p_value = kruskal(*groups)
    kruskal_rows.append({"feature": feature, "H_statistic": h_stat, "p_value": p_value})

kruskal_df = pd.DataFrame(kruskal_rows)
if len(kruskal_df) > 0:
    kruskal_df["p_bonferroni"] = (kruskal_df["p_value"] * len(kruskal_df)).clip(upper=1.0)
    kruskal_df.to_csv(kruskal_csv, index=False, encoding="utf-8-sig")

if SCIPOSTHOCS_AVAILABLE and len(kruskal_df) > 0:
    dunn_summary_rows = []
    for feature in feature_cols:
        matched = kruskal_df.loc[kruskal_df["feature"] == feature, "p_bonferroni"]
        if len(matched) == 0 or matched.values[0] >= 0.05: continue
        dunn_result = sp.posthoc_dunn(work_df, val_col=feature, group_col="cluster", p_adjust="bonferroni")
        dunn_result.to_csv(os.path.join(dunn_dir, f"dunn_{feature}_K{final_k}.csv"), encoding="utf-8-sig")


# 7. XAI 분석 (Random Forest + SHAP)
print("=== XAI (Random Forest + SHAP) 분석 시작 ===")
if SHAP_AVAILABLE:
    # 스케일링 전 원본 변수를 사용하여 의학적 해석을 용이하게 함
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, work_df["cluster"])
    
    # Feature Importance 저장
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    importances.to_csv(rf_importance_csv, index=False, encoding="utf-8-sig")
    print("\n[Random Forest Feature Importance]")
    print(importances.to_string(index=False))

    # SHAP 분석
    # === 이 부분을 기존 7번 파트 SHAP 시각화 코드와 교체하세요 ===
    
    # SHAP 분석
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # 1. 기존 막대그래프 (필요하면 남겨두고 필요 없으면 지우셔도 됩니다)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar - All Clusters)")
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, "shap_summary_bar.png"), dpi=300)
    plt.close()

    # 2. 새로 추가: 올려주신 사진처럼 예쁜 '점 그래프 (Beeswarm)' 생성
    # 다중 분류이므로 각 군집(Cluster)별로 따로 그려야 에러가 안 나고 예쁘게 나옵니다.
    # 2. 예쁜 점 그래프 (Beeswarm) 추가 생성
    classes = sorted(work_df["cluster"].unique())
    
    for c_idx in classes:
        plt.figure(figsize=(8, 6))
        
        # SHAP 라이브러리 버전에 따른 데이터 구조 분기 처리 (에러 해결 핵심)
        if isinstance(shap_values, list):
            # 구버전 SHAP: 리스트 형태일 때
            class_shap_vals = shap_values[c_idx]
        else:
            # 신버전 SHAP: 3차원 Numpy 배열 형태일 때 [samples, features, classes]
            class_shap_vals = shap_values[:, :, c_idx]
        
        # 추출한 해당 군집의 SHAP 값으로 점 그래프 생성
        shap.summary_plot(class_shap_vals, X, show=False) 
        
        plt.title(f"SHAP Summary (Dot) - Cluster {c_idx}")
        plt.tight_layout()
        
        save_path = os.path.join(xai_dir, f"shap_summary_dot_Cluster{c_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        plt.close()
        print(f"Cluster {c_idx} SHAP Dot Plot 저장 완료: {save_path}")
else:
    print("SHAP 모듈이 없어 XAI 분석 생략.\n")


# 8. 시각화 (Boxplot, K-지표, 차원축소 2D, 3D 산점도)
print("=== 시각화 이미지 저장 중 ===")
# 차원 축소 2D 시각화 함수
def plot_2d(x_col, y_col, title, filename):
    plt.figure(figsize=(8, 6))
    for c in sorted(work_df["cluster"].unique()):
        sub = work_df[work_df["cluster"] == c]
        label_name = f"Cluster {c}: {sub['cluster_type'].iloc[0]}" if final_k == 3 else f"Cluster {c}"
        plt.scatter(sub[x_col], sub[y_col], alpha=0.7, label=label_name, s=30)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=300)
    plt.close()

plot_2d("PC1", "PC2", f"PCA of K-Means (K={final_k})", f"kmeans_pca_K{final_k}.png")
plot_2d("tSNE1", "tSNE2", f"t-SNE of K-Means (K={final_k})", f"kmeans_tsne_K{final_k}.png")
if UMAP_AVAILABLE:
    plot_2d("UMAP1", "UMAP2", f"UMAP of K-Means (K={final_k})", f"kmeans_umap_K{final_k}.png")

# 3D 산점도
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for c in sorted(work_df["cluster"].unique()):
    sub = work_df[work_df["cluster"] == c]
    label_name = f"Cluster {c}: {sub['cluster_type'].iloc[0]}" if final_k == 3 else f"Cluster {c}"
    ax.scatter(sub["arterial_diameter"], sub["venous_diameter"], sub["loop_diameter"],
               alpha=0.7, s=40, label=label_name, edgecolors='w', linewidth=0.5)
ax.set_xlabel("Arterial Diameter (px)", labelpad=10)
ax.set_ylabel("Venous Diameter (px)", labelpad=10)
ax.set_zlabel("Loop Diameter (px)", labelpad=10)
ax.view_init(elev=20, azim=45) 
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, f"kmeans_3d_scatter_K{final_k}.png"), dpi=400)
plt.close()


# 9. 결과 CSV 저장 및 대표 이미지 복사
work_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
cluster_summary.to_csv(summary_csv, encoding="utf-8-sig")

# 이미지 복사 로직 (기존과 동일)
def resolve_image_path(img_dir, fname):
    fname = str(fname).strip()
    exact = os.path.join(img_dir, fname)
    if os.path.exists(exact): return exact
    stem, _ = os.path.splitext(fname)
    for ext2 in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        cand = os.path.join(img_dir, stem + ext2)
        if os.path.exists(cand): return cand
    matches = glob.glob(os.path.join(img_dir, "**", stem + ".*"), recursive=True)
    if matches: return matches[0]
    return None

rep_rows = []
for c in sorted(work_df["cluster"].unique()):
    cluster_idxs = work_df[work_df["cluster"] == c].index.tolist()
    if not cluster_idxs: continue
    
    center = X_scaled[cluster_idxs].mean(axis=0)
    dist_list = sorted([(idx, np.linalg.norm(X_scaled[idx] - center)) for idx in cluster_idxs], key=lambda x: x[1])[:5]
    
    cluster_dir = os.path.join(rep_copy_root, f"kmeans_cluster_{c}")
    os.makedirs(cluster_dir, exist_ok=True)
    
    for idx, dist in dist_list:
        fname = work_df.loc[idx, "filename"]
        img_path = resolve_image_path(image_dir, fname)
        copied, reason = ("YES", "") if img_path else ("NO", "not_found")
        if img_path: shutil.copy2(img_path, os.path.join(cluster_dir, os.path.basename(img_path)))
        
        rep_rows.append({
            "cluster": c, "filename": fname, "distance": round(dist, 4), "copied": copied
        })

pd.DataFrame(rep_rows).to_csv(rep_csv, index=False, encoding="utf-8-sig")

print("=========================================")
print("분석 및 저장 완료!")
print(f"- 군집/차원축소 결과 CSV: {out_csv}")
print(f"- SHAP 요약 플롯 (XAI): {os.path.join(xai_dir, 'shap_summary_bar.png')}")
print(f"- 2D/3D 시각화 이미지: {result_dir} 폴더 내 확인")
print("=========================================")