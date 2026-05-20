"""
Hierarchical Clustering + Kruskal-Wallis + Effect Size + Dunn Post-hoc + Boxplot
Times New Roman figure version
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
from scipy.stats import kruskal

# ============================================================
# 0. 논문용 공통 폰트 설정
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11


def apply_times_new_roman(ax, label_size=13, tick_size=11, title_size=15, bold_label=True):
    """
    matplotlib axis에 Times New Roman 폰트 강제 적용
    """
    fontweight = "bold" if bold_label else "normal"

    ax.xaxis.label.set_fontname("Times New Roman")
    ax.yaxis.label.set_fontname("Times New Roman")
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight(fontweight)
    ax.yaxis.label.set_fontweight(fontweight)

    ax.title.set_fontname("Times New Roman")
    ax.title.set_fontsize(title_size)
    ax.title.set_fontweight("bold")

    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(tick_size)

    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(tick_size)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontname("Times New Roman")
            text.set_fontsize(tick_size)


# Dunn post-hoc test용 패키지
try:
    import scikit_posthocs as sp
    SCIPOSTHOCS_AVAILABLE = True
except ImportError:
    SCIPOSTHOCS_AVAILABLE = False
    print("scikit-posthocs가 설치되어 있지 않음.")
    print("Dunn post-hoc test를 수행하려면 아래 명령어로 설치:")
    print("pip install scikit-posthocs")
    print()


# ============================================================
# 1. 기본 경로 설정
# ============================================================

csv_path = "limb_total.csv"
image_dir = r"E:\MTL_dataset\image"

result_dir = "hierarchical_results"
os.makedirs(result_dir, exist_ok=True)

score_csv = os.path.join(result_dir, "limb_total_hierarchical_k_scores.csv")
pca_loading_csv = os.path.join(result_dir, "limb_total_hierarchical_pca_loadings.csv")


# ============================================================
# 2. CSV 로드 및 컬럼 확인
# ============================================================

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


# ============================================================
# 3. 필요한 컬럼만 사용
# ============================================================

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


# ============================================================
# 4. Feature Engineering
# ============================================================

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


# ============================================================
# 5. 표준화
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 6. PCA
# ============================================================

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


# ============================================================
# 7. Hierarchical Clustering K 비교
# ============================================================

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

best_k = int(
    score_df
    .sort_values("silhouette", ascending=False)
    .iloc[0]["K"]
)

print(f"silhouette 기준 best K: {best_k}")
print()


# ============================================================
# 8. 최종 Hierarchical Clustering
# ============================================================

final_k = best_k

print(f"최종 사용 K: {final_k}")
print()

hier_model = AgglomerativeClustering(
    n_clusters=final_k,
    linkage="ward"
)

work_df["hier_cluster_raw"] = hier_model.fit_predict(X_scaled)


# ============================================================
# 9. K=3일 경우 군집 의미 재매핑
# ============================================================

if final_k == 3:
    raw_summary = (
        work_df
        .groupby("hier_cluster_raw")[feature_cols]
        .mean()
    )

    venous_dominant_raw = raw_summary["va_ratio"].idxmax()

    remaining_summary = raw_summary.drop(index=venous_dominant_raw)

    enlarged_raw = remaining_summary["avg_diameter3"].idxmax()
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
    work_df["hier_cluster"] = work_df["hier_cluster_raw"]
    work_df["hier_cluster_type"] = "not_mapped"

    print("final_k가 3이 아니므로 KMeans-style label mapping은 적용하지 않음.")
    print("K=4 이상은 별도 의미 해석 필요.")
    print()


# ============================================================
# 10. 저장 경로 설정
# ============================================================

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

kruskal_csv = os.path.join(
    result_dir,
    f"limb_total_hierarchical_kruskal_K{final_k}.csv"
)

dunn_summary_csv = os.path.join(
    result_dir,
    f"limb_total_hierarchical_dunn_summary_K{final_k}.csv"
)

dunn_dir = os.path.join(
    result_dir,
    f"dunn_posthoc_K{final_k}"
)

boxplot_dir = os.path.join(
    result_dir,
    f"cluster_boxplots_K{final_k}"
)

os.makedirs(dunn_dir, exist_ok=True)
os.makedirs(boxplot_dir, exist_ok=True)


# ============================================================
# 11. 군집별 요약 통계
# ============================================================

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


# ============================================================
# 12. Kruskal-Wallis Test
# ============================================================

cluster_col = "hier_cluster"

kruskal_rows = []

print("=== Kruskal-Wallis test: 군집 간 feature 차이 검정 ===")

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

    n_total = len(work_df)
    k_groups = work_df["hier_cluster"].nunique()

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


# ============================================================
# 13. Dunn Post-hoc Test
# ============================================================

dunn_summary_rows = []

if len(kruskal_df) > 0 and SCIPOSTHOCS_AVAILABLE:
    print("=== Dunn post-hoc test: 군집 간 쌍별 비교 ===")

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
            group_col="hier_cluster",
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

        clusters = sorted(work_df["hier_cluster"].unique())

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


# ============================================================
# 14. Cluster-wise Boxplot
# ============================================================

print("=== Cluster-wise boxplot 저장 ===")

for feature in feature_cols:
    fig, ax = plt.subplots(figsize=(7, 5))

    sorted_clusters = sorted(work_df["hier_cluster"].unique())

    data_to_plot = [
        work_df.loc[work_df["hier_cluster"] == c, feature].dropna()
        for c in sorted_clusters
    ]

    label_names = []

    for c in sorted_clusters:
        if final_k == 3:
            c_type = (
                work_df.loc[work_df["hier_cluster"] == c, "hier_cluster_type"]
                .iloc[0]
            )
            label_names.append(f"C{c}\n{c_type}")
        else:
            label_names.append(f"Cluster {c}")

    ax.boxplot(
        data_to_plot,
        tick_labels=label_names,
        showfliers=False
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by cluster")
    ax.grid(alpha=0.3, linestyle=":")

    apply_times_new_roman(ax)

    plt.tight_layout()

    save_path = os.path.join(
        boxplot_dir,
        f"boxplot_{feature}_K{final_k}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Boxplot 저장: {os.path.abspath(save_path)}")

print()


# ============================================================
# 15. 군집 해석용 통합 요약표 저장
# ============================================================

cluster_summary_mean = (
    work_df
    .groupby("hier_cluster")[feature_cols]
    .mean()
)

cluster_summary_std = (
    work_df
    .groupby("hier_cluster")[feature_cols]
    .std()
)

cluster_summary_count = (
    work_df
    .groupby("hier_cluster")
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

if "hier_cluster_type" in work_df.columns:
    cluster_type_map = (
        work_df
        .groupby("hier_cluster")["hier_cluster_type"]
        .first()
    )
    cluster_interpretation["cluster_type"] = cluster_type_map

cluster_interpretation = cluster_interpretation.reset_index()

cluster_interpretation_csv = os.path.join(
    result_dir,
    f"limb_total_hierarchical_cluster_interpretation_K{final_k}.csv"
)

cluster_interpretation.to_csv(
    cluster_interpretation_csv,
    index=False,
    encoding="utf-8-sig"
)

print("=== 군집 해석용 통합 요약표 ===")
print(cluster_interpretation.round(3).to_string(index=False))
print(f"군집 해석용 통합 요약표 저장: {os.path.abspath(cluster_interpretation_csv)}")
print()


# ============================================================
# 16. CSV 저장
# ============================================================

work_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
cluster_summary.to_csv(summary_csv, encoding="utf-8-sig")
cluster_std.to_csv(std_csv, encoding="utf-8-sig")

print(f"샘플별 Hierarchical 결과 저장: {os.path.abspath(out_csv)}")
print(f"군집 요약표 저장: {os.path.abspath(summary_csv)}")
print(f"군집 표준편차 저장: {os.path.abspath(std_csv)}")
print(f"K 비교 점수 저장: {os.path.abspath(score_csv)}")
print()


# ============================================================
# 17. K별 평가 지표 그래프 저장
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(score_df["K"], score_df["silhouette"], marker="o")
ax.axvline(final_k, linestyle=":", linewidth=1.5)
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Silhouette score")
ax.set_title("Hierarchical Clustering: Silhouette Score by K")
ax.grid(alpha=0.3, linestyle=":")
apply_times_new_roman(ax)
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_silhouette_by_k.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(score_df["K"], score_df["calinski_harabasz"], marker="o")
ax.axvline(final_k, linestyle=":", linewidth=1.5)
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Calinski-Harabasz score")
ax.set_title("Hierarchical Clustering: Calinski-Harabasz Score by K")
ax.grid(alpha=0.3, linestyle=":")
apply_times_new_roman(ax)
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_calinski_by_k.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(score_df["K"], score_df["davies_bouldin"], marker="o")
ax.axvline(final_k, linestyle=":", linewidth=1.5)
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Davies-Bouldin score")
ax.set_title("Hierarchical Clustering: Davies-Bouldin Score by K")
ax.grid(alpha=0.3, linestyle=":")
apply_times_new_roman(ax)
plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_davies_by_k.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 18. PCA 시각화
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for c in sorted(work_df["hier_cluster"].unique()):
    sub = work_df[work_df["hier_cluster"] == c]

    ax.scatter(
        sub["PC1"],
        sub["PC2"],
        alpha=0.7,
        label=f"Cluster {c}",
        s=30
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
ax.set_title(f"PCA of Hierarchical Clustering Results (K={final_k})")
ax.legend()
ax.grid(alpha=0.3, linestyle=":")

apply_times_new_roman(ax)

plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, f"hierarchical_pca_cluster_K{final_k}.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 19. Dendrogram
# ============================================================

linked = linkage(X_scaled, method="ward")

# ------------------------------------------------------------
# 19-1. Full dendrogram
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 7))

dendrogram(
    linked,
    no_labels=True,
    color_threshold=None,
    ax=ax
)

# 제목을 쓰고 싶으면 아래 주석 해제
# ax.set_title("Hierarchical Clustering Dendrogram (Ward linkage)")

ax.set_xlabel("Samples")
ax.set_ylabel("Linkage Distance")

apply_times_new_roman(
    ax,
    label_size=13,
    tick_size=11,
    title_size=15,
    bold_label=True
)

ax.grid(False)

plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_dendrogram_full.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ------------------------------------------------------------
# 19-2. Truncated dendrogram
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

dendrogram(
    linked,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True,
    color_threshold=None,
    ax=ax
)

# 제목을 쓰고 싶으면 아래 주석 해제
# ax.set_title("Truncated Hierarchical Dendrogram")

ax.set_xlabel("Samples")
ax.set_ylabel("Linkage Distance")

apply_times_new_roman(
    ax,
    label_size=13,
    tick_size=11,
    title_size=15,
    bold_label=True
)

ax.grid(False)

plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "hierarchical_dendrogram_truncated.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 20. 대표 이미지 경로 찾기 함수
# ============================================================

def resolve_image_path(image_dir, filename):
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


# ============================================================
# 21. 대표 이미지 선택 및 복사
# ============================================================

rep_rows = []
missing_files = []

os.makedirs(rep_copy_root, exist_ok=True)

for c in sorted(work_df["hier_cluster"].unique()):
    cluster_idxs = work_df[work_df["hier_cluster"] == c].index.tolist()

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


# ============================================================
# 22. 최종 출력
# ============================================================

print("=========================================")
print("Hierarchical Clustering 분석 완료")
print("K-Means와 동일한 의미 체계로 label 정렬 완료")

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