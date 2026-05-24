"""
K-Means Clustering + Kruskal-Wallis + Effect Size + Dunn Post-hoc + Boxplot
(폰트: Times New Roman, 군집별 고정 색상 적용, 축 이름 볼드 처리)
"""

import os
os.environ["OMP_NUM_THREADS"] = "6"

import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 폰트를 Times New Roman으로 전역 설정
plt.rcParams['font.family'] = 'Times New Roman'
# 폰트 깨짐 방지용 마이너스 기호 설정
plt.rcParams['axes.unicode_minus'] = False

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
csv_path = "limb_total.csv" # 각 직경이 정리되어있는 csv 파일
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


# 군집별 고정 색상 팔레트 (Cluster 0: 파랑, 1: 주황, 2: 초록)
CLUSTER_COLORS = {
    0: '#1f77b4', 
    1: '#ff7f0e', 
    2: '#2ca02c'
}


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

for k in k_list:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20 
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

    venous_dominant_raw = raw_summary["va_ratio"].idxmax()
    remaining_summary = raw_summary.drop(index=venous_dominant_raw)
    enlarged_raw = remaining_summary["avg_diameter3"].idxmax()
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

    # 오타 수정 (Blanced -> Balanced)
    work_df["cluster_type"] = work_df["cluster"].map(
        {
            0: "Asymmetric",
            1: "Dilated",
            2: "Balanced",
        }
    )

    print("=== K-Means raw label → 의미 기반 label mapping ===")
    print(label_map)
    print()

else:
    work_df["cluster"] = work_df["cluster_raw"]
    work_df["cluster_type"] = "not_mapped"
    print("final_k가 3이 아니므로 의미 기반 label mapping은 적용하지 않음.")
    print()

# 군집별 요약 통계
cluster_counts = work_df["cluster"].value_counts().sort_index()
cluster_summary = work_df.groupby("cluster")[feature_cols].mean().round(3)
cluster_std = work_df.groupby("cluster")[feature_cols].std().round(3)

if final_k == 3:
    cluster_type_summary = (
        work_df
        .groupby(["cluster", "cluster_type"])
        .size()
        .reset_index(name="count")
    )


# Kruskal-Wallis Test
cluster_col = "cluster"
kruskal_rows = []

for feature in feature_cols:
    groups = []
    group_sizes = {}

    for c in sorted(work_df[cluster_col].unique()):
        values = work_df.loc[work_df[cluster_col] == c, feature].dropna()
        groups.append(values)
        group_sizes[f"cluster_{c}_n"] = len(values)

    if len(groups) < 2 or any(len(g) == 0 for g in groups):
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

kruskal_df = pd.DataFrame(kruskal_rows)

if len(kruskal_df) > 0:
    kruskal_df["p_bonferroni"] = (kruskal_df["p_value"] * len(kruskal_df)).clip(upper=1.0)
    kruskal_df["significant_bonferroni_0.05"] = (kruskal_df["p_bonferroni"] < 0.05)
    n_total = len(work_df)
    k_groups = work_df["cluster"].nunique()
    if n_total > k_groups:
        kruskal_df["epsilon_squared"] = ((kruskal_df["H_statistic"] - k_groups + 1) / (n_total - k_groups)).clip(lower=0, upper=1)
    else:
        kruskal_df["epsilon_squared"] = np.nan
    kruskal_df = kruskal_df.sort_values("p_value")
    kruskal_df.to_csv(kruskal_csv, index=False, encoding="utf-8-sig")


# Dunn Post-hoc Test
dunn_summary_rows = []

if len(kruskal_df) > 0 and SCIPOSTHOCS_AVAILABLE:
    for feature in feature_cols:
        matched = kruskal_df.loc[kruskal_df["feature"] == feature, "p_bonferroni"]
        if len(matched) == 0: continue
        p_val = matched.values[0]
        if p_val >= 0.05: continue

        dunn_result = sp.posthoc_dunn(work_df, val_col=feature, group_col="cluster", p_adjust="bonferroni")
        dunn_csv = os.path.join(dunn_dir, f"dunn_posthoc_{feature}_K{final_k}.csv")
        dunn_result.to_csv(dunn_csv, encoding="utf-8-sig")

        clusters = sorted(work_df["cluster"].unique())
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                try: pair_p = dunn_result.loc[c1, c2]
                except KeyError: pair_p = dunn_result.loc[str(c1), str(c2)]
                
                dunn_summary_rows.append({
                    "feature": feature, "comparison": f"cluster_{c1}_vs_cluster_{c2}",
                    "p_adjusted": pair_p, "significant_0.05": pair_p < 0.05
                })

    dunn_summary_df = pd.DataFrame(dunn_summary_rows)
    if len(dunn_summary_df) > 0:
        dunn_summary_df.to_csv(dunn_summary_csv, index=False, encoding="utf-8-sig")


# Cluster-wise Boxplot
for feature in feature_cols:
    plt.figure(figsize=(7, 5))
    sorted_clusters = sorted(work_df["cluster"].unique())
    data_to_plot = [work_df.loc[work_df["cluster"] == c, feature].dropna() for c in sorted_clusters]
    label_names = []
    
    for c in sorted_clusters:
        if final_k == 3:
            c_type = work_df.loc[work_df["cluster"] == c, "cluster_type"].iloc[0]
            label_names.append(f"C{c}\n{c_type}")
        else:
            label_names.append(f"Cluster {c}")

    box = plt.boxplot(data_to_plot, labels=label_names, showfliers=False, patch_artist=True)
    
    # 박스플롯에도 군집 색상 적용
    for patch, c in zip(box['boxes'], sorted_clusters):
        patch.set_facecolor(CLUSTER_COLORS.get(c, '#cccccc'))
        patch.set_alpha(0.6)

    # 폰트 볼드 처리 추가
    plt.xlabel("Cluster", fontweight='bold')
    plt.ylabel(feature, fontweight='bold')
    
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(boxplot_dir, f"boxplot_{feature}_K{final_k}.png"), dpi=300)
    plt.close()


# 군집 해석용 통합 요약표 저장
cluster_summary_mean = work_df.groupby("cluster")[feature_cols].mean()
cluster_summary_std = work_df.groupby("cluster")[feature_cols].std()
cluster_summary_count = work_df.groupby("cluster").size().rename("n")

cluster_interpretation = cluster_summary_mean.copy()
cluster_interpretation.columns = [f"{c}_mean" for c in cluster_interpretation.columns]
for col in feature_cols:
    cluster_interpretation[f"{col}_std"] = cluster_summary_std[col]
cluster_interpretation["n"] = cluster_summary_count

if "cluster_type" in work_df.columns:
    cluster_type_map = work_df.groupby("cluster")["cluster_type"].first()
    cluster_interpretation["cluster_type"] = cluster_type_map

cluster_interpretation = cluster_interpretation.reset_index()
cluster_interpretation_csv = os.path.join(result_dir, f"limb_total_kmeans_cluster_interpretation_K{final_k}.csv")
cluster_interpretation.to_csv(cluster_interpretation_csv, index=False, encoding="utf-8-sig")

# 결과 CSV 저장
work_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
cluster_summary.to_csv(summary_csv, encoding="utf-8-sig")
cluster_std.to_csv(std_csv, encoding="utf-8-sig")


# =====================================================================
# 2차원 PCA 시각화 (군집별 고정 색상 + 축 이름 볼드 처리)
# =====================================================================
plt.figure(figsize=(8, 6))

for c in sorted(work_df["cluster"].unique()):
    sub = work_df[work_df["cluster"] == c]

    if final_k == 3:
        label_name = sub["cluster_type"].iloc[0]
        legend_label = f"Cluster {c}: {label_name}"
    else:
        legend_label = f"Cluster {c}"

    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        alpha=0.7,
        label=legend_label,
        s=30,
        color=CLUSTER_COLORS.get(c, None) # 고정 색상 적용
    )

# 폰트 볼드 처리 추가
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontweight='bold', fontsize=14)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontweight='bold', fontsize=14)
plt.legend()
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
pca_save_path = os.path.join(result_dir, f"kmeans_pca_cluster_K{final_k}.png")
plt.savefig(pca_save_path, dpi=400)
plt.close()


# =====================================================================
# 3차원 산점도 시각화 (안전한 파일 저장 방식으로 수정)
# =====================================================================
print("=== 3D Scatter plot 저장 (여러 각도) ===")

# 테스트해 볼 시야각 리스트
view_angles = [
    (20, 45),   # 기존 각도
    (20, 135),  # 반대편에서 본 각도
    (30, 225),  # 위에서 비스듬히 본 각도 (이전에 가장 좋았던 각도)
    (30, -60)   # 기본 각도
]

for elev, azim in view_angles:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for c in sorted(work_df["cluster"].unique()):
        sub = work_df[work_df["cluster"] == c]

        if final_k == 3 and "cluster_type" in work_df.columns:
            label_name = sub["cluster_type"].iloc[0]
            legend_label = f"Cluster {c}: {label_name}"
        else:
            legend_label = f"Cluster {c}"

        ax.scatter(
            sub["arterial_diameter"],
            sub["venous_diameter"],
            sub["loop_diameter"],
            alpha=0.8,       
            s=25,            
            label=legend_label,
            edgecolors='w',
            linewidth=0.4,
            color=CLUSTER_COLORS.get(c, None)
        )

    # 폰트 볼드 처리 추가
    ax.set_xlabel("Arterial Diameter (px)", labelpad=10, fontweight='bold', fontsize=14)
    ax.set_ylabel("Venous Diameter (px)", labelpad=10, fontweight='bold', fontsize=14)
    ax.set_zlabel("Loop Diameter (px)", labelpad=10, fontweight='bold', fontsize=14)

    # 시야각 적용
    ax.view_init(elev=elev, azim=azim) 

    plt.legend()
    plt.tight_layout()
    
    # [수정] 파일 저장 에러 방지: 
    # 1. 절대경로(abspath) 사용
    # 2. 보이지 않는 특수문자 방지
    safe_filename = f"kmeans_3d_scatter_K{final_k}_elev{elev}_azim{azim}.png"
    save_path_3d = os.path.abspath(os.path.join(result_dir, safe_filename))
    
    try:
        # 3. 메모리 충돌 방지를 위해 해상도(dpi)를 논문 권장치인 300으로 조정
        plt.savefig(save_path_3d, dpi=300)
        print(f"저장 성공: {safe_filename}")
    except Exception as e:
        print(f"저장 실패 ({safe_filename}): {e}")
        
    # 4. 다음 반복문을 위해 현재 생성된 그래프 메모리를 완전히 초기화(close)
    plt.close(fig)

print("축이 볼드 처리된 4가지 방향의 3D Scatter plot 저장이 완료되었습니다!")
print()


