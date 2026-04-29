"""
Subject-level Cluster Composition Analysis

목적:
- 전체 1388장에 대해 이미 부여된 cluster label을 사용
- 각 피험자별로 Cluster 0, 1, 2가 몇 개씩 나오는지 확인
- 피험자별 cluster 비율 계산
- dominant cluster 확인
- stacked bar plot / heatmap 저장

전제:
- hierarchical_results/limb_total_hierarchical_clustered_K3.csv 파일이 존재해야 함
- 해당 CSV 안에 filename, hier_cluster 컬럼이 있어야 함
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result_dir = "hierarchical_results"

clustered_csv = os.path.join(
    result_dir,
    "limb_total_hierarchical_clustered_K3.csv"
)

subject_result_dir = os.path.join(
    result_dir,
    "subject_level_analysis"
)

os.makedirs(subject_result_dir, exist_ok=True)


# 저장 파일 경로
subject_count_csv = os.path.join(
    subject_result_dir,
    "subject_cluster_count.csv"
)

subject_ratio_csv = os.path.join(
    subject_result_dir,
    "subject_cluster_ratio.csv"
)

subject_summary_csv = os.path.join(
    subject_result_dir,
    "subject_cluster_summary.csv"
)

stacked_bar_png = os.path.join(
    subject_result_dir,
    "subject_cluster_composition_stacked_bar.png"
)

heatmap_png = os.path.join(
    subject_result_dir,
    "subject_cluster_ratio_heatmap.png"
)

dominant_bar_png = os.path.join(
    subject_result_dir,
    "subject_dominant_cluster_bar.png"
)


# =========================================================
# 1. 군집화 결과 CSV 불러오기
# =========================================================

df = pd.read_csv(clustered_csv, encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

print("현재 컬럼명:")
print(df.columns.tolist())
print()


# =========================================================
# 2. 필수 컬럼 확인
# =========================================================

required_cols = ["filename", "hier_cluster"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

print(f"전체 샘플 수: {len(df)}")
print()


# =========================================================
# 3. filename에서 subject ID 추출
# =========================================================
# 예:
# p18_det_076_crop_0.png -> p18
# p1_det_103.png         -> p1

df["subject"] = (
    df["filename"]
    .astype(str)
    .str.lower()
    .str.extract(r"(p\d+)", expand=False)
)

# subject 추출 실패 확인
missing_subject = df["subject"].isna().sum()

if missing_subject > 0:
    print(f"subject ID를 추출하지 못한 샘플 수: {missing_subject}")
    print("추출 실패 예시:")
    print(df.loc[df["subject"].isna(), "filename"].head(10).tolist())
    print()
    raise ValueError("filename에서 subject ID 추출 실패 샘플이 있음. filename 형식 확인 필요.")

print("추출된 피험자 수:")
print(df["subject"].nunique())
print()

print("피험자 목록:")
print(sorted(df["subject"].unique(), key=lambda x: int(x.replace("p", ""))))
print()


# =========================================================
# 4. cluster 번호 의미 지정
# =========================================================
# 네가 K-Means와 맞춘 번호 체계 기준:
# Cluster 0 = 세정맥 우세 비대칭형
# Cluster 1 = 전반적 확장형
# Cluster 2 = 소직경 균형형

cluster_name_map = {
    0: "venous_dominant_asymmetric",
    1: "global_enlargement",
    2: "small_balanced",
}

df["cluster_type"] = df["hier_cluster"].map(cluster_name_map)

# 혹시 0,1,2 외 cluster가 있는지 확인
unknown_cluster = df["cluster_type"].isna().sum()

if unknown_cluster > 0:
    print("주의: cluster_name_map에 없는 cluster 번호가 있음.")
    print("실제 cluster 번호:")
    print(sorted(df["hier_cluster"].unique()))
    print()
    print("K=4 결과가 섞였거나 label mapping이 안 된 상태일 수 있음.")
    print("일단 unknown으로 처리함.")
    df["cluster_type"] = df["cluster_type"].fillna("unknown")


# =========================================================
# 5. 피험자별 cluster 개수 계산
# =========================================================

subject_cluster_count = pd.crosstab(
    df["subject"],
    df["hier_cluster"]
)

# subject 순서 p1, p2, p3 ... 형태로 정렬
subject_order = sorted(
    subject_cluster_count.index,
    key=lambda x: int(x.replace("p", ""))
)

subject_cluster_count = subject_cluster_count.loc[subject_order]

# cluster 번호 순서 정렬
subject_cluster_count = subject_cluster_count.reindex(
    sorted(subject_cluster_count.columns),
    axis=1
)

print("=== 피험자별 cluster 개수 ===")
print(subject_cluster_count)
print()


# =========================================================
# 6. 피험자별 cluster 비율 계산
# =========================================================

subject_cluster_ratio = subject_cluster_count.div(
    subject_cluster_count.sum(axis=1),
    axis=0
)

print("=== 피험자별 cluster 비율 ===")
print(subject_cluster_ratio.round(3))
print()


# =========================================================
# 7. 피험자별 dominant cluster 계산
# =========================================================

subject_summary = subject_cluster_ratio.copy()

subject_summary["dominant_cluster"] = subject_cluster_ratio.idxmax(axis=1)
subject_summary["dominant_ratio"] = subject_cluster_ratio.max(axis=1)
subject_summary["n_samples"] = subject_cluster_count.sum(axis=1)

subject_summary["dominant_cluster_type"] = (
    subject_summary["dominant_cluster"]
    .map(cluster_name_map)
)

print("=== 피험자별 dominant cluster ===")
print(subject_summary.round(3))
print()


# =========================================================
# 8. CSV 저장
# =========================================================

subject_cluster_count.to_csv(
    subject_count_csv,
    encoding="utf-8-sig"
)

subject_cluster_ratio.to_csv(
    subject_ratio_csv,
    encoding="utf-8-sig"
)

subject_summary.to_csv(
    subject_summary_csv,
    encoding="utf-8-sig"
)

print(f"피험자별 cluster 개수 저장: {os.path.abspath(subject_count_csv)}")
print(f"피험자별 cluster 비율 저장: {os.path.abspath(subject_ratio_csv)}")
print(f"피험자별 요약 저장: {os.path.abspath(subject_summary_csv)}")
print()


# =========================================================
# 9. 피험자별 cluster 비율 Stacked Bar Plot
# =========================================================

plt.figure(figsize=(14, 6))

ax = subject_cluster_ratio.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 6)
)

plt.xlabel("Subject")
plt.ylabel("Cluster proportion")
plt.title("Subject-level Cluster Composition")

legend_labels = []

for c in subject_cluster_ratio.columns:
    if c in cluster_name_map:
        legend_labels.append(f"Cluster {c}: {cluster_name_map[c]}")
    else:
        legend_labels.append(f"Cluster {c}: unknown")

plt.legend(
    legend_labels,
    title="Cluster type",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3, linestyle=":")
plt.tight_layout()

plt.savefig(stacked_bar_png, dpi=300)
plt.show()

print(f"Stacked bar plot 저장: {os.path.abspath(stacked_bar_png)}")
print()


# =========================================================
# 10. 피험자별 cluster 비율 Heatmap
# =========================================================

heatmap_data = subject_cluster_ratio.copy()

plt.figure(figsize=(8, 10))

plt.imshow(
    heatmap_data.values,
    aspect="auto"
)

plt.xticks(
    ticks=np.arange(heatmap_data.shape[1]),
    labels=[f"Cluster {c}" for c in heatmap_data.columns]
)

plt.yticks(
    ticks=np.arange(heatmap_data.shape[0]),
    labels=heatmap_data.index
)

plt.colorbar(label="Proportion")

plt.xlabel("Cluster")
plt.ylabel("Subject")
plt.title("Heatmap of Subject-level Cluster Proportions")

# 각 cell에 숫자 표시
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        value = heatmap_data.iloc[i, j]
        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.tight_layout()
plt.savefig(heatmap_png, dpi=300)
plt.show()

print(f"Heatmap 저장: {os.path.abspath(heatmap_png)}")
print()


# =========================================================
# 11. Dominant cluster 분포 Bar Plot
# =========================================================

dominant_counts = (
    subject_summary["dominant_cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 5))

plt.bar(
    [str(c) for c in dominant_counts.index],
    dominant_counts.values
)

plt.xlabel("Dominant cluster")
plt.ylabel("Number of subjects")
plt.title("Number of Subjects by Dominant Cluster")
plt.grid(axis="y", alpha=0.3, linestyle=":")

for i, v in enumerate(dominant_counts.values):
    plt.text(
        i,
        v + 0.05,
        str(v),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(dominant_bar_png, dpi=300)
plt.show()

print(f"Dominant cluster bar plot 저장: {os.path.abspath(dominant_bar_png)}")
print()


# =========================================================
# 12. 결과 해석용 출력
# =========================================================

print("=========================================")
print("Subject-level cluster composition 분석 완료")
print("Cluster 번호 의미:")
print("0 = 세정맥 우세 비대칭형")
print("1 = 전반적 확장형")
print("2 = 소직경 균형형")
print()
print("해석 방향:")
print("- subject_cluster_count.csv: 피험자별 cluster 개수")
print("- subject_cluster_ratio.csv: 피험자별 cluster 비율")
print("- subject_cluster_summary.csv: dominant cluster 및 샘플 수")
print("- stacked bar plot: 23명 전체의 cluster 구성 비율 확인")
print("- heatmap: 피험자별 cluster 우세 패턴 확인")
print("=========================================")