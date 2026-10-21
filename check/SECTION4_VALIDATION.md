# 제4절 직경 검증 실행 순서

**현재 선정 자료는 범주별 48장, 총 144장이다.** 현재 실행기·CSV·GT 복사본은
`check/SECTION4_PA_144.md`를 따른다. 아래 150장 설명은 이전 선정 자료의
작업 기록이다.

## 현재 150장: 개선 알고리즘 점검

`C:\Users\USER\Desktop\Pa_capillary`의 U/tortuous/twist 각 50장에 대해
**원본 RGB에서 승인된 수동 측정**과 해당 환자가 학습에서 제외된 fold의
**MTL 예측 분할 마스크·U/D**를 짝짓는다. GT 분할 마스크는 자동 직경의
최종 입력으로 사용하지 않는다.

프로젝트 폴더의 `check/run_section4_pa_150.cmd --dry-run`으로 150장의
OOF 대응을 확인하고, `check/run_section4_pa_150.cmd`로 두 분지 직경을
산출한다. 기본 출력은 `check/section4_oof_limb_pa_150_v3.csv`이며
기존 `section4_oof_limb_pa_150.csv`와 v2를 덮어쓰지 않는다. 동일한 v3 파일을
다시 만들 때만 `--overwrite`를 추가한다. 최종 선정 MTL 모델이 달라지면
`--run-dir <완료된_5fold_실행_경로> --output <새_CSV_경로>`를 함께 지정해
현재 코드 점검 결과를 보존한다.

개선된 코드는 끝점이 닫힌 분지에서도 아래쪽 경로를 찾고, 두 중심선이
같은 국소 단면에 있으면서 그 사이에 배경 간격이 있는 지점만 측정한다.
각 분지의 **최대 유효 국소 직경**을 기록한다. `Branch*_Ray_Valid_Points`가
5 미만이거나 EDT fallback을 사용한 경우 두 직경은 기록하되
`Status=Uncertain`으로 표시한다. 측정 불가인 경우 `Status=Failed`와
`Reason`을 남긴다. 일치도 계산은 `Success`만 포함하며, 전체 선택 수와
Uncertain/Failed 수는 별도 보고한다. EDT fallback의 선 끝점은 표시용
근사점으로 실제 분할 경계점이라고 해석하지 않는다. CSV의
`Algorithm_SHA256`은 analyzer, core algorithm, utils, exporter 소스의
해시이고 `Mask_SHA256`은 각 예측 마스크의 해시다. 한 분지 중심선 길이가 상대
분지의 1/3 미만인 경우에도 누락된 분지 가능성이 있어 `Uncertain`으로
표시하며 `Branch_Path_Balance`를 CSV에 보존한다.

수동 라벨을 원본에서 재검수한 뒤 다음을 실행한다. 마지막 명령은 전체 및
형태별 Pearson, ICC 절대일치·일관성, MAE, Bland–Altman 지표와 환자 단위
bootstrap 신뢰구간을 저장한다.

```powershell
$py = 'C:\ProgramData\anaconda3\python.exe'
$run = 'C:\Users\USER\Desktop\Multi-Task-Learning\results_unet_mtl_heatmap_bce_dice_5849db92da83'
& $py .\check\section4_validation.py `
  --selection-dir 'C:\Users\USER\Desktop\Pa_capillary' `
  --oof-root $run `
  --manual-csv '.\label\outputs\section4_review_pa_150\manual_review_150.csv' `
  --limb-csv '.\check\section4_oof_limb_pa_150_v3.csv' `
  --output-dir '.\check\section4_validation_pa_150_v3'
```

검토 필요 사례와 수동값 차이가 큰 사례의 원본·예측 마스크·측정선을 한 화면에
표시하려면 다음을 실행한다. `review_queue.csv`에는 검토 이유와 차이가 저장된다.
이 목록은 오류 재검토용이며 같은 사례로 임계값을 조정한 뒤 독립 검증 성능으로
해석하지 않는다.

```powershell
& $py .\check\section4_qc_review.py `
  --manual-csv '.\label\outputs\section4_review_pa_150\manual_review_150.csv' `
  --algorithm-csv '.\check\section4_oof_limb_pa_150_v3.csv' `
  --selection-dir 'C:\Users\USER\Desktop\Pa_capillary' `
  --output-dir '.\check\section4_qc_pa_150_v3' --top-n 10
```

이 150장의 일부 사례는 알고리즘 오류를 찾는 데 이미 사용했다. 따라서
변경 후 같은 150장과의 일치도는 **탐색적 내부 검증**으로 기술한다.
독립적인 최종 성능을 주장하려면 별도 환자 단위 평가 세트가 필요하다.
작은/큰 분지 직경은 정지영상에서의 형태학적 세동맥측/세정맥측 추정이며
혈류 방향으로 확인한 생리학적 A/V 판정은 아니다.

## 이전 51장 작업 기록

아래 명령과 상태 설명은 과거 51장 검토 당시의 기록이다. 현재 150장 측정에는
위의 v3 실행 순서와 결과 파일을 사용한다.

이 폴더의 스크립트는 `Capillary_limb` 프로젝트에서 사용한다. 원본 직경 알고리즘은
`limb/analyzer.py`, 과거 수동 분지 라벨은 `analysis/total/image/passivity.csv`이다.
새 검수 결과와 OOF 직경 추출 결과는 **별도 파일**에 저장하며 원본을 바꾸지 않는다.

## 1. 수동 측정 검수

프로젝트 루트 `C:\Users\USER\Desktop\Capillary_limb`에서 실행한다.

```powershell
$py = 'C:\ProgramData\anaconda3\python.exe'
& $py .\label\section4_manual_review.py --check --blind-loop --output-csv '.\label\outputs\section4_review\manual_review_blind_loop.csv'
& $py .\label\section4_manual_review.py --blind-loop --output-csv '.\label\outputs\section4_review\manual_review_blind_loop.csv'
```

기존 51개 **분지 측정선**은 검수 전 `pending` 초안으로 표시된다.
`--blind-loop`는 MTL 학습 U/D 라벨을 가리고 원본 RGB 영상에서 루프를 새로
측정한다. 모든 영상에서 두 분지와 루프의 선을 확인한 뒤 승인하거나 측정 불가
사유를 기록한다. 화면에 자동 예측은 표시하지 않는다. 자세한 조작과 측정 기준은
`label/SECTION4_MANUAL_REVIEW.md`에 있다.

## 2. MTL 환자별 외부 fold 예측 확인

완료된 MTL 실행을 지정해 선택 51장의 예측 마스크와 U/D가 각각 **한 번만**
`outer_test`에서 생성됐는지 확인한다. 아래 실행은 코드 점검용으로 이미 완료된
U-Net MTL 실행이다. 최종 채택 모델이 다르면 해당 모델의 완료된 5-fold 실행
경로로 바꾼다.

```powershell
$run = 'C:\Users\USER\Desktop\Multi-Task-Learning\results_unet_mtl_heatmap_bce_dice_5849db92da83'
& $py .\check\section4_oof_limb_export.py --run-dir $run --dry-run
```

이 실행은 현재 51장, 환자 17명, fold별 11/12/8/13/7장의 일대일 OOF 대응을
확인했다. 51장은 MTL의 전체 1,399장 안에 있으므로 전체 자료로 다시 학습한
단일 모델의 출력은 직경 성능 평가에 사용하지 않는다.

## 3. 예측 마스크에서 양쪽 분지 직경 추출

`scipy`, `scikit-image`, `opencv-python`, `pandas`, `numpy`가 설치된 **기존 분석
환경**에서 아래 명령을 실행한다. 추출 CSV에는 성공뿐 아니라 실패 영상도
포함하고, fold, 예측 마스크 경로·해시, 알고리즘 소스 해시를 기록한다.

```powershell
& $analysisPy .\check\section4_oof_limb_export.py --run-dir $run --output '.\check\section4_oof_limb_final.csv'
```

`$analysisPy`는 직경 분석에 사용한 Python 실행 파일 경로로 지정한다. 현재
확인 가능한 Anaconda 기본 환경에는 OpenCV가 없어 전체 직경 추출을 실행하지
못했고, OOF 무결성 확인(`--dry-run`)까지만 수행했다.

## 4. 수동 대 자동 일치도

```powershell
& $py .\check\section4_validation.py `
  --selection-dir 'C:\Users\USER\Desktop\analysis\image' `
  --oof-root $run `
  --manual-csv '.\label\outputs\section4_review\manual_review_blind_loop.csv' `
  --limb-csv '.\check\section4_oof_limb_final.csv' `
  --output-dir '.\check\section4_validation_final'
```

출력은 영상별 결합·검증 범위 CSV와 루프/가는 분지/굵은 분지 각각의 Pearson
상관계수, ICC(A,1) 절대일치, ICC(C,1) 일관성, Bland–Altman 평균차·일치한계,
MAE이다. 전체와 U자형/굴곡형/꼬임형을 구분한다. 95% 신뢰구간은 **환자 단위
군집 bootstrap**으로 계산한다. 검수 `pending`, 측정 불가, 알고리즘 실패는
짝지어진 통계에서 제외되되, 영상별 범위 표에는 남는다. 수동 승인 전에 나오는
수치는 최종 결과로 해석하지 않는다.

가는/굵은 분지는 정지영상에서 상대 직경으로 **형태학적 추정**한
세동맥측/세정맥측이다. 이 통계는 두 폭의 일치도이며 혈류 방향에 의한 생리학적
A/V 판정 정확도를 입증하지 않는다. 51장은 환자별 OOF **내부 검증의 형태별
하위 분석**이다. 같은 51장의 수동 결과를 보고 알고리즘의 임계값을 더 조정하면
독립적인 최종 검증으로 표현하지 않는다.
