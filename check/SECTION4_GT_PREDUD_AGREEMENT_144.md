# 제4절 GT 분할 + MTL 예측 U/D 직경 일치도 (144장)

## 비교 대상

- 수동 기준: 원본 RGB 영상에서 측정하고 승인한 `label/outputs/section4_review_pa_144/manual_review_144.csv` (144장, 20명).
- 자동 입력: 같은 144장의 **복사된 GT 분할 마스크**와 환자별 바깥 테스트(OOF) **MTL 예측 U/D**.
- 분지 직경: 현행 `limb/analyzer.py`의 중심선·국소 수직 ray-casting/EDT 분석. 가는/굵은 분지는 각 영상의 두 폭을 정렬한 형태학적 구분이다.
- 루프 직경: 예측 U/D 사이의 유클리드 거리. 이 값은 GT 분할 마스크를 사용하지 않는다.

## 현재 예비 실행 결과

현재 완료된 OOF MTL 실행 `5849db92da83`으로 코드를 검증했다. 아직 학습 중인 최종 선정 모델의 결과가 아니다.

| 측정값 | 짝지은 영상 | Pearson r | ICC(C,1) 일관성 | ICC(A,1) 절대일치 | MAE (px) |
|---|---:|---:|---:|---:|---:|
| 루프 | 144 | 0.7387 | 0.7305 | 0.6308 | 5.8130 |
| 가는 분지 | 132 | 0.7677 | 0.7676 | 0.7674 | 2.8440 |
| 굵은 분지 | 132 | 0.7995 | 0.7945 | 0.7855 | 3.0547 |

GT 마스크 분석의 상태는 Success 132장, Uncertain 12장, Failed 0장이다. 분지 ICC와 Pearson에는 수동 승인 및 Success인 132장(20명)을 사용했다. 루프에는 분지 상태와 무관하게 예측 U/D가 유효한 144장(20명)을 사용했다. 전체 및 U/tortuous/twist별 수치와 환자 단위 bootstrap 95% 신뢰구간은 `section4_gt_predud_agreement_pa_144_v1_precheck/section4_gt_predud_agreement.csv`에 저장한다.

같은 폴더의 `section4_gt_predud_pairs.csv`는 144장을 모두 보존하며 영상별 수동값·자동값·측정 상태·오차를 담는다. `section4_gt_predud_coverage.csv`에는 형태별 Success/Uncertain/Failed 수를, `section4_gt_predud_provenance.json`에는 입력 파일과 해시를 담는다.

## 최종 MTL 실행으로 다시 계산

최종 모델의 **환자 단위 OOF** 실행 폴더가 완성되면 PowerShell에서 다음 두 명령을 실행한다. `<최종_실행_폴더>`를 실제 경로로 바꾸고, 기존 예비 결과와 다른 출력 이름을 유지한다.

```powershell
$py = 'C:\ProgramData\anaconda3\python.exe'
$check = 'C:\Users\USER\Desktop\Capillary_limb\check'
$run = '<최종_실행_폴더>'
$gtCsv = "$check\section4_gt_limb_pa_144_final.csv"
$outDir = "$check\section4_gt_predud_agreement_pa_144_final"

& $py "$check\section4_gt_limb_export.py" --run-dir $run --output $gtCsv
& $py "$check\section4_gt_predud_agreement.py" --run-dir $run --gt-export-csv $gtCsv --output-dir $outDir
```

두 스크립트는 선정 144장, OOF 폴드와 예측 U/D, GT 복사본·원본 해시, 수동 측정선과 저장 직경값의 대응을 검사한다. 기존 출력이 있으면 덮어쓰기 옵션 없이 중단한다. 학습 데이터와 원본 수동 CSV는 수정하지 않는다.

이 실험은 GT 마스크를 사용해 분할 오차를 줄여 본 **보조 기하학 평가**다. 실제 MTL 전체 파이프라인의 결과는 예측 분할 마스크와 예측 U/D를 함께 입력한 결과로 별도 보고한다. 현재 144장은 알고리즘 점검에도 사용되었으므로 예비 수치를 독립적인 최종 검증 성능으로 해석하지 않는다.
