# 전체 DINOv3 계측값 군집화

## 현재 기본 분석

전체 1,397장에 저장된 DINOv3 outer-test 예측 마스크와 U/D 좌표를 적용했다. 사용자의 전체 데이터 포함 방침에 따라 **측정값이 있는 Success 1,302장과 Uncertain 78장, 총 1,380장을 모두 기본 군집에 포함한다.**

논문 방법에 맞춰 확장한 결과 폴더는 `C:\Users\USER\Desktop\Multi-Task-Learning\outputs\dinov3_full_clustering_20261009\analysis_paper_measured_legacy5_k3_um`이다. `cluster_assignments_all.csv`에는 전체 1,397행을 보존한다. 직경값을 구하지 못한 Failed 17장은 군집 번호를 비워 두며 사유를 기록한다. 결측 직경을 0이나 평균값으로 채우지 않는다. 이전 결과 폴더는 별도로 보존한다.

`dinov3_full_clustering.py`의 기본 `--qc-policy`는 `measured`이다. 이전에 작성한 `Success` 1,302장만의 결과는 품질 경고 포함 여부를 비교하는 보조 분석으로 보관한다.

## Uncertain의 뜻

`Uncertain`은 **두 분지의 직경은 계산됐지만 측정 과정에서 품질 경고가 발생했다**는 뜻이다. DINOv3의 확률이나 군집의 소속 확률을 나타내는 상태가 아니다.

기존 측정 내보내기 코드가 사용하는 경고 기준은 다음과 같다.

- 짧은 분지 경로의 점 개수가 긴 분지의 1/3 미만이다. 한쪽 경로가 불완전하게 추출됐을 가능성을 표시한다.
- 어느 한 분지에서 유효한 직경 측정 후보점이 5개 미만이다.
- 경계 탐색 대신 EDT 거리변환으로 직경을 대체했다.

이번 Uncertain 78장은 경로 길이 불균형 76장, 유효 후보점 부족 3장으로 구성된다. 두 경고가 겹치는 1장이 있고 EDT 대체 사례는 없다. 이 표시만으로 실제 직경이 틀렸다고 확정할 수 없으므로 기본 분석에 포함하고 경고를 보존한다.

Failed 17장은 두 분지 경로 추출 실패 9장과 유효한 분지 측정 결과 없음 8장이다. 이 영상들까지 직경 기반 군집 번호를 부여하려면 먼저 누락된 직경을 측정해야 한다.

## 데이터와 특성

- 원본: `D:\MTL_dataset\image`, 1,397장, 23명.
- 모델: 완료된 5-fold U-Net MTL DINOv3 ViT-B/16 실행 `dbc65ae36d7e`.
- 모델 결과: `C:\Users\USER\Desktop\논문 결과\results_unet_mtl_dinov3_vitb16_finetune_heatmap_dbc65ae36d7e`.
- 전체 계측 CSV: `Multi-Task-Learning/outputs/dinov3_full_clustering_20261009/canonical_measurements_1397.csv`.
- 계측 실행 코드: `Multi-Task-Learning/check/export_dinov3_full_measurements.py`. 기존 `section4_oof_limb_export.py`와 `limb/analyzer.py`를 재사용한다.

각 영상은 자신의 outer-test fold 예측을 사용한다. 원본 영상, 학습 명단, master 좌표, fold별 좌표 및 마스크의 대응을 검증했다. 기존 141장 결과와 겹치는 영상의 상태·사유·해시 및 30개 수치 열은 전체 계측 결과와 동일했다.

기본 `legacy5` 입력은 가는 분지 직경, 굵은 분지 직경, U/D 사이 루프 직경, 굵은/가는 분지 비율, 세 직경 평균이다. **1 px = 0.55 µm**로 직경을 실제 환산한 뒤 모든 변수를 표준화한다. 직경·평균 직경 열은 `_um`, 직경비는 무차원이다. 가는/굵은 분지는 폭으로 정렬한 값이며 해부학적 동맥/정맥 판정은 아니다. 비율과 평균은 세 직경에서 계산된 파생 변수이다. 영상 임베딩이나 U/tortuous/crossing 형상 표지는 입력에 사용하지 않았다.

## 기본 결과

1,380장, 기존 5변수, K=3의 K-means 군집 크기는 **788 / 127 / 465장**이다. 군집 번호는 각 방법에서 평균 직경이 작은 순서로 정렬한다. 서로 다른 변수 설정이나 방법의 같은 번호가 같은 군집을 뜻하지는 않는다.

| 방법 | K=2 silhouette | K=3 silhouette |
| --- | ---: | ---: |
| K-means | 0.3696 | 0.3732 |
| Ward 계층 군집 | 0.3392 | 0.3487 |

이 포함 기준에서 논문과 동일한 K=2~6 중 K=3의 silhouette가 가장 높다. 최종 K는 `--k`로 지정하며 기본값은 3이다. Elbow plot도 함께 검토하도록 제공하고 자동으로 최적 elbow를 확정하지 않는다. 점수 차이가 작으므로 뚜렷하게 분리된 세 유형이 확정됐다는 의미는 아니다. 이전 Success-only 분석에서 K=2 점수가 높았다는 설명은 1,302장만 사용한 결과에 해당한다.

기본 결과 폴더의 `cluster_feature_summary.csv`는 군집별 측정값, `patient_cluster_composition.csv`는 환자별 영상 구성, `representative_images.csv`는 군집 중심과 가까운 영상 목록이다. 아래 파일로 논문 분석 결과를 확인한다. 모든 그림은 PNG 300 dpi, TIFF 600 dpi(LZW), PDF로 저장한다.

그림 스타일은 기존 수동/자동 직경 비교 그림과 동일하게 통일했다. 축·눈금은 **DejaVu Sans 11 pt**, 범례는 **9 pt**이며 군집 색은 `#2474a6`, `#c66b20`, `#418652`이다. 흰 배경, 네 면의 검정 테두리 0.8 pt, 격자 alpha 0.2를 사용하고 **모든 그림의 제목은 제거**한다. 공통 설정은 `dinov3_cluster_plot_style.py`에 있다. PCA·UMAP·t-SNE·SHAP의 점 면적은 22에서 **12 pt²**, 점 테두리는 **0.4 pt**로 줄였다. 전체 계산 실행의 단위·스타일은 `run_metadata.json`에 기록한다. 저장 결과에서 그림만 다시 그린 경우 그 이력은 `figure_style_metadata.json`에 기록한다.

| 확인할 내용 | 파일 |
| --- | --- |
| 전체 1,397장 포함/제외 상태와 군집 | `cluster_assignments_all.csv` |
| K=2~6 silhouette·SSE | `cluster_k_scores.csv`, `silhouette_by_k.png`, `elbow_by_k.png` |
| PCA·UMAP·t-SNE 군집 분포 | `kmeans_pca.png`, `kmeans_umap.png`, `kmeans_tsne.png` |
| UMAP·t-SNE 나란히 비교 | `kmeans_umap_tsne_comparison.png` |
| 영상별 PCA·UMAP 및 t-SNE 좌표 | `projection_coordinates.csv`, `tsne_coordinates.csv` |
| PCA 설명분산 및 주성분 계수 | `pca_explained_variance.csv`, `pca_components.csv` |
| 원자료의 표준화 값 및 변환 모수 | `standardized_features.csv`, `scaler_parameters.csv` |
| 군집별 형태 특징 분포 | `interpretation/feature_boxplots.png`, `interpretation/feature_descriptives_by_cluster.csv` |
| Kruskal-Wallis 및 epsilon-squared | `interpretation/kruskal_wallis_effect_sizes.csv` |
| RF 대리모델의 학습 자료 군집 재현도 | `interpretation/rf_training_confusion_matrix.png`, `interpretation/rf_training_classification_report.csv` |
| SHAP 특징별·군집별 기여도 | `interpretation/shap_grouped_bar.png`, `interpretation/shap_beeswarm_cluster_0.png`~`2.png` |
| SHAP 수치 원본 및 대응 정보 | `interpretation/shap_values.npz`, `interpretation/shap_mean_abs_by_class.csv` |
| 실행 설정·버전·코드 해시 | `run_metadata.json`, `projection_metadata.json`, `interpretation/interpretation_metadata.json` |

## µm 환산과 단위

- 기존 수동/자동 직경 비교의 사용자 지정 보정값 **0.55 µm/px**를 재사용한다. 근거는 `C:\Users\USER\Desktop\Multi-Task-Learning\outputs\predicted_mask_measurements_dinov3_141_20261008_run3\provenance.json`이다. 전체 1,397장과 예측 좌표는 동일한 512×512 좌표계다.
- 실제 루프·두 분지 직경, 가는/굵은 분지, 평균 직경을 환산해 `_um` 열에 저장한다. 군집 요약과 분포 통계, RF 입력·SHAP 특징값도 µm를 사용한다. 원본 `_px` 열과 이미지 표시용 좌표는 추적을 위해 보존한다.
- 평균 직경은 세 값이 모두 있어야 계산한다. 분지 측정 실패 17장의 평균 직경은 결측으로 유지한다.
- 직경비, z-score, PCA·UMAP 축, silhouette, 표준화 공간 SSE, 효과크기, RF 확률 및 SHAP 기여도는 길이 값이 아니므로 µm를 붙이거나 0.55를 곱하지 않는다.
- 보정값을 바꿀 때는 `--um-per-pixel`로 명시한다. 같은 양의 환산값을 곱한 후 z-score를 적용하므로 군집 공간은 수치 오차 범위에서 같다.

## 논문 방법과 실행 설정

1. **군집화:** `StandardScaler`의 z-score 표준화 후 5차원 특징에서 K-means를 적용한다. `init="k-means++"`, `n_init=20`, `random_state=42`이다. K=2~6을 평가하고 각 모델의 SSE(`inertia_`)와 silhouette를 저장한다. Ward는 보조 비교이다.
2. **시각화:** 동일한 표준화 특징에 PCA와 UMAP을 각각 적용한다. 군집 라벨은 차원축소 학습에 전달하지 않고 그림의 색에만 사용한다. PCA/UMAP 좌표는 K-means와 Ward 그림에서 공유한다. UMAP은 `n_components=2`, `n_neighbors=15`, `min_dist=0.1`, `spread=1`, `metric="euclidean"`, `init="spectral"`, `random_state=42`, `transform_seed=42`, `n_jobs=1`이다. PCA의 PC1·PC2 누적 설명분산은 약 87.01%이다.
3. **특징 비교:** 실제 Kruskal-Wallis H와 p를 산출한다. 주 효과크기는 기존 `stat_statistics.py`와 같은 rank epsilon-squared **H/(N−1)**이다. 다른 구형 스크립트에서 쓴 **(H−k+1)/(N−k)**는 rank eta-squared 대안으로 별도 저장한다. 원 p와 5개 특징에 대한 Bonferroni 보정 p를 함께 저장한다. 별도의 사후 쌍별 검정이나 고정 유의성 별표는 사용하지 않는다.
4. **대리모델:** 기존 K-means 라벨을 의사 목표로 RF 100개 트리(`random_state=42`)를 학습한다. 입력은 같은 특징의 원 단위 값이다. 평가는 같은 학습 자료에서의 군집 재현도(resubstitution fidelity)이며 새 환자나 임상 분류 성능이 아니다.
5. **SHAP:** `TreeExplainer`의 `tree_path_dependent`, `model_output="raw"`를 명시한다. 이 RF에서 출력은 군집별 확률이다. 모든 포함 영상에 대한 평균 절대 SHAP 값과 군집별 beeswarm을 저장한다. 클래스 순서는 `rf.classes_`로 검증하고 기준값+SHAP 합이 RF 확률과 일치하는지 확인한다. `shap_values.npz`는 `[영상, 특징, 클래스]` 순서와 각 축의 이름을 보존한다.

### 본문 수정 시 확인할 부분

- 이번 DINOv3 실행은 **1,397장 중 17장 측정 실패, 1,380장 분석**이다. 이전 분석의 ‘11장 제외·1,386장 사용’을 그대로 사용하지 않는다. Uncertain 78장은 현재 분석에 포함했다.
- 기존 arterial/venous 열은 실제로 두 분지의 굵기를 정렬한 값이다. 해부학적 판정 근거가 없다면 ‘굵기 기준 세동맥측·세정맥측 대리지표’ 등으로 정의해야 한다. 현재 그림은 Thin/Wide branch로 표시하며 직경 단위는 µm이다.
- 표준화는 특징의 분산을 맞춘다. 평균과 비율이 같은 직경에서 파생되므로 5개 특징의 독립성이나 동등한 기여도를 보장하는 것은 아니다.
- 같은 군집 입력 특징의 Kruskal-Wallis 결과는 군집의 독립적 유효성 검증이 아닌 분포 설명이다. 여러 모세혈관이 같은 환자에서 나왔으므로 이 p값에는 환자 내 의존성 보정도 포함되지 않는다. SHAP 역시 대리모델의 설명으로 해석한다.

설정 참고: [UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html), [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html), [SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html), [rank epsilon/eta 구현](https://raw.githubusercontent.com/easystats/effectsize/main/R/rank_ANOVA.R).

기존 `analysis_success_*` 폴더는 Success-only 품질 및 변수 선택 비교용이다. Success-only K=3에서 직접 측정 3변수와 기존 5변수 군집의 ARI는 0.326으로, 변수 선택의 영향이 있다.

## 좌표 품질 감사

`keypoint_error_audit.csv`는 저장된 GT U/D 길이와 예측 길이를 비교한 사후 감사다. 이 감사로 기본 군집에서 영상을 추가 제외하지 않았다. 전체 절대오차 중앙값은 4.11 px, 95백분위는 11.20 px이다. 분지 분석기의 Success도 DINOv3 U/D 좌표 정확도까지 보증하지는 않는다.

## 다시 실행하기

이미 계측한 전체 CSV를 사용하므로 아래 군집화 명령만 실행하면 된다. 기존 출력은 덮어쓰지 않으므로 매번 새 출력 폴더를 지정한다. `younguk` 환경에 UMAP·SHAP 의존성 설치와 전체 실행을 확인했다. 다른 환경에서는 이 디렉터리의 `requirements_dinov3_clustering.txt`를 사용한다.

```powershell
$py = 'C:\Users\USER\.conda\envs\younguk\python.exe'
$root = 'C:\Users\USER\Desktop\Multi-Task-Learning\outputs\dinov3_full_clustering_20261009'
& $py -X utf8 -B 'C:\Users\USER\Desktop\Capillary_limb\cluster\dinov3_full_clustering.py' --input "$root\canonical_measurements_1397.csv" --output-dir "$root\analysis_paper_measured_legacy5_k3_um_run2" --k 3 --max-k 6 --qc-policy measured --feature-set legacy5 --um-per-pixel 0.55
```

군집화는 입력 CSV와 같은 폴더의 `provenance.json`을 검증한다. `--qc-policy success`는 경고 영상 제외에 따른 보조 비교에만 사용한다. 기존 `cluster_main.py`, `K-means.py`, `Hierarchical.py`는 예전 경로와 CSV를 가정하므로 이번 실행의 진입점은 `dinov3_full_clustering.py`이다.

### 계산된 결과에서 그림만 다시 저장하기

아래 명령은 저장된 군집·PCA·UMAP 좌표와 SHAP 값을 읽어 그림을 다시 저장한다. CSV·NPZ·모델 파일이 실행 전후 바이트 단위로 같은지 확인한다. 그림과 스타일 메타데이터는 기존 폴더에서 갱신한다.

```powershell
& 'C:\Users\USER\.conda\envs\younguk\python.exe' -X utf8 -B 'C:\Users\USER\Desktop\Capillary_limb\cluster\redraw_dinov3_cluster_figures.py' --analysis-dir 'C:\Users\USER\Desktop\Multi-Task-Learning\outputs\dinov3_full_clustering_20261009\analysis_paper_measured_legacy5_k3_um'
```

## t-SNE 추가 탐색

동일한 1,380장의 표준화된 5차원 특징으로 t-SNE도 계산한다. 기존 K-means/Ward 군집 번호는 그림의 색에만 사용한다. µm 원자료를 직접 t-SNE에 넣거나 UMAP 좌표를 재차 축소하지 않는다. t-SNE 축은 무차원이다.

- 설정: `n_components=2`, `perplexity=30`, `init="pca"`, `learning_rate="auto"`(이번 실행에서 50), `early_exaggeration=12`, `max_iter=1000`, `metric="euclidean"`, `random_state=42`, `method="barnes_hut"`, `angle=0.5`, `n_jobs=1`; 수치 라이브러리 스레드 수 1.
- perplexity는 그림을 보기 전에 기본값 30으로 정했다. 시각적 분리가 좋아지는 설정을 골라 군집 타당성을 주장하지 않는다. 최대 반복 수는 수렴을 보장하는 값이 아니다.
- 좌표와 실제 KL divergence·반복 수·패키지 버전·입력 해시는 `tsne_coordinates.csv`, `tsne_metadata.json`에 저장한다. 기존 계산 결과와 그림이 보존됐는지는 `tsne_validation.json`에서 확인한다.
- 단일 그림: `kmeans_tsne.png/pdf/tiff`, `ward_tsne.png/pdf/tiff`. 비교 그림: `kmeans_umap_tsne_comparison.png/pdf/tiff`. 제목 없이 기존 폰트·색상·작은 점을 사용한다.
- 이후 `dinov3_full_clustering.py` 전체 실행에도 t-SNE가 포함되며 `--tsne-perplexity`로 값을 명시할 수 있다. 기존 결과에 t-SNE가 아직 없다면 아래 명령으로 추가한다. 이미 추가된 결과는 덮어쓰지 않는다.

```powershell
& 'C:\Users\USER\.conda\envs\younguk\python.exe' -X utf8 -B 'C:\Users\USER\Desktop\Capillary_limb\cluster\add_tsne_dinov3.py' --analysis-dir 'C:\Users\USER\Desktop\Multi-Task-Learning\outputs\dinov3_full_clustering_20261009\analysis_paper_measured_legacy5_k3_um' --perplexity 30
```

UMAP 또는 t-SNE에서 색이 겹치는 정도는 기존 5차원 공간의 군집 타당성을 직접 판단하는 지표가 아니다. t-SNE는 국소적 이웃 관계를 강조하므로 그림의 섬 간 거리와 크기를 실제 군집 간 차이로 해석하지 않는다. [scikit-learn t-SNE 안내](https://scikit-learn.org/1.7/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne)
