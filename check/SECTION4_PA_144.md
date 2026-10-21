# 제4절 현재 선정 자료: 144장

현재 평가 대상은 `C:\Users\USER\Desktop\Pa_capillary`의 `U`, `tortuous`,
`twist` 각 48장(총 144장, 20명)이다. 파일명과 영상 해시의 중복이 없다.
이전 150장 파일은 과거 선정 기준의 기록으로 보존한다.

## 활성 파일

- 수동 직경: `C:\Users\USER\Desktop\Capillary_limb\label\outputs\section4_review_pa_144\manual_review_144.csv`
- 수동 측정 재검수: `C:\Users\USER\Desktop\Capillary_limb\label\run_pa_144.cmd`
- GT 분할 복사본: `C:\Users\USER\Desktop\Capillary_limb\check\section4_gt_masks_pa_144\U|tortuous|twist`
- GT 대응 명세: `C:\Users\USER\Desktop\Capillary_limb\check\section4_gt_masks_pa_144\gt_mask_manifest_144.csv`
- 예측 분할 기반 OOF 직경: `C:\Users\USER\Desktop\Capillary_limb\check\section4_oof_limb_pa_144_v4.csv`
- 예측 직경 재실행: `C:\Users\USER\Desktop\Capillary_limb\check\run_section4_pa_144.cmd`
- 예비 검증: `C:\Users\USER\Desktop\Capillary_limb\check\section4_validation_pa_144_v4_precheck`
- 오류 검토 그림·목록: `C:\Users\USER\Desktop\Capillary_limb\check\section4_qc_pa_144_v4`

`label/run_pa_144.cmd --check`에서 144장 모두 승인·여섯 점 완비를 확인했다.
수동 CSV는 이전 150장 CSV의 선정 144행을 복사했고, 좌표·직경값·승인
기록을 변경하지 않았다. `p1_det_084.png`만 폴더 이동에 따라 `shape`을
`tortuous`에서 `twist`로 바꿨다. 해당 영상은 시각적으로 뚜렷한 분지 교차가
보이지 않으므로 논문 범주 확정 전에 형태를 재검수하는 편이 좋다.

이전 150장에서 제외된 파일명은 다음 여섯 개다:

`p11_det_013_crop_0.png`, `p22_det_112_crop_2.png`,
`p9_det_038_crop_0.png`, `p7_det_005_crop_3.png`,
`p7_det_007_crop_1.png`, `p7_det_008_crop_1.png`.

GT TIFF 144개는 학습 데이터 `D:\MTL_dataset\label`을 수정하지 않고 기존
복사본에서 새 폴더로 복사했다. 각 복사본의 SHA-256이 학습 데이터 원본과
일치하며 모두 512×512이다. 원본 TIFF 형식은 유지했다. 알고리즘에 입력할 때는
예측 마스크와 같은 `>127` 기준으로 이진화한다.

OOF 예측 직경은 144행 중 `Success` 130, `Uncertain` 14, `Failed` 0이다.
예비 비교는 수동 재검수와 최종 MTL 모델 적용 전의 내부 코드 점검으로만
해석한다. 최종 모델이 달라지면 `run_section4_pa_144.cmd --run-dir <새_5fold_결과>
--output <새_CSV>`로 기존 파일을 보존해 다시 추출한다.
