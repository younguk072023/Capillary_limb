# 제4절 144장 직경 알고리즘 실행 기록

## 입력과 목적

- 선정 영상: `C:\Users\USER\Desktop\Pa_capillary`의 U, tortuous, twist 각 48장, 총 144장(20명).
- 수동 기준: 원본 RGB 영상에서 그은 세 개의 직경선이 담긴 `C:\Users\USER\Desktop\Capillary_limb\label\outputs\section4_review_pa_144\manual_review_144.csv`.
- 주 분석: 환자 단위 바깥 테스트(OOF) MTL 예측 마스크와 예측 U/D로 직경 측정.
- 보조 분석: 같은 예측 U/D와 같은 분석기를 유지하고 마스크만 복사된 GT TIFF로 교체. GT 분할은 직경의 정답이 아니다.

## 결과 파일

- 예측 마스크 직경: `C:\Users\USER\Desktop\Capillary_limb\check\section4_oof_limb_pa_144_v4.csv`
- GT 마스크 직경: `C:\Users\USER\Desktop\Capillary_limb\check\section4_gt_limb_pa_144_v1.csv`
- 짝지은 비교: `C:\Users\USER\Desktop\Capillary_limb\check\section4_mask_input_pa_144_v1\section4_mask_input_pairs.csv`
- 요약: 같은 폴더의 `section4_mask_input_summary.csv`, `section4_mask_input_coverage.csv`, `section4_mask_input_provenance.json`
- 실행 코드: `check\section4_gt_limb_export.py`, `check\section4_mask_paired_compare.py`

## 현재 예비 결과

| 마스크 입력 | Success | Uncertain | Failed |
|---|---:|---:|---:|
| MTL 예측 | 130 | 14 | 0 |
| GT 복사본 | 132 | 12 | 0 |

두 입력 모두 Success인 126장(19명)에서 같은 원본영상 수동선과 비교했다. 각 영상의 두 분지 측정값을 작은 순서와 큰 순서로 정렬해 짝지었다.

| 직경 | 예측 마스크 MAE (px) | GT 마스크 MAE (px) | 예측 오차−GT 오차 평균 (px) | 환자 단위 95% bootstrap 구간 (px) |
|---|---:|---:|---:|---:|
| 가는 분지 | 2.708 | 2.823 | −0.115 | −0.397 ~ 0.273 |
| 굵은 분지 | 2.804 | 3.058 | −0.254 | −0.660 ~ 0.216 |

GT 마스크로 교체해도 이 자료에서는 수동선과의 오차가 줄지 않았다. 두 차이의 구간은 모두 0을 포함한다. 루프 직경은 마스크를 사용하지 않고 동일한 예측 U/D의 거리로 계산하므로 두 실행에서 완전히 같다. 기존 예비 비교에서 루프는 144장 MAE 5.813 px, ICC(A,1) 0.631이다.

## 해석과 확인할 사례

- 이 비교는 **분할 마스크 입력의 민감도**를 확인한다. 예측 U/D는 고정되어 있어 키포인트 오류는 두 실행 모두에 남는다.
- `p4_det_074_crop_0.png`, `p4_det_086_crop_0.png`의 기존 QC 그림에서는 알고리즘이 분지 아래쪽의 넓어진 부분에서 최대폭을 고르고, 수동 측정선은 위쪽에 있다. 측정 위치 차이가 오차의 원인일 수 있어 수동선과 알고리즘의 해부학적 측정 구간을 재검수해야 한다. 그림은 `check\section4_qc_pa_144_v4`에 있다.
- 가는/굵은 분지는 직경 순서로 정한 **형태학적 추정**이다. 영상만으로 실제 혈류 방향에 따른 세동맥/세정맥을 확인한 결과로 쓰면 안 된다.
- 이 144장은 알고리즘 점검과 개선에도 사용되었으므로 현재 수치는 내부 예비 결과다. 최종 모델과 독립 환자 자료를 정한 후 본 결과를 다시 산출한다. 수동선도 최종 재검수한다.

GT 복사본 144개는 원본 GT와 SHA-256이 일치한다. 두 실행의 이미지, 폴드, 예측 U/D, 분석기 SHA-256이 동일한 것은 비교 코드에서 검사한다. 코드 단위 테스트 10개가 통과했다.
