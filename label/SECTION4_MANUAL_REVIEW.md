# 제4절 직경 수동 검수

**현재 선정 자료는 각 범주 48장, 총 144장이다.** 현재 측정 기록은
`label/outputs/section4_review_pa_144/manual_review_144.csv`이고,
재검수할 때 `label/run_pa_144.cmd`를 실행한다. 아래 150장 설명은 이전
선정 자료의 작업 기록이다.

144장 간편 화면에서는 오른쪽에 현재 U-D, B1, B2 선의 길이(px)를 바로
표시한다. 점을 고르고 영상을 클릭하면 좌표와 길이가 즉시 CSV에 저장된다.
여섯 점과 세 길이가 모두 유효하면 자동으로 `approved`가 된다. 왼쪽/오른쪽
방향키나 Previous/Next 버튼으로 이동할 수 있고, 이동·창 닫기 때도 저장한다.
이전 영상으로 돌아가 점을 다시 찍으면 해당 좌표와 길이로 갱신된다.
간편 화면에서는 승인/측정불가/대기 저장 버튼을 표시하지 않는다. 아래의
150장 및 기존 51장 실행 방식은 기존 검수 화면을 유지한다.

`run_pa_144.cmd`는 통합 `manual_review_144.csv`와 함께
`manual_review_144_u.csv`, `manual_review_144_tortuous.csv`,
`manual_review_144_crossing.csv`를 같은 폴더에 저장한다. 유형별 CSV는
통합 CSV의 해당 영상 48행을 그대로 담는다. Crossing은 원래 데이터의
`twist` 유형이므로 Crossing CSV 안의 `shape` 값은 `twist`로 유지한다.
점 수정·영상 이동·창 닫기 때 네 파일을 다시 저장한다. 작업을 재개할
때는 통합 CSV를 읽는다.

## 새로 선정한 150장 수동 측정

`C:\Users\USER\Desktop\Pa_capillary`의 `U`, `tortuous`, `twist` 폴더를
한 화면에서 순서대로 검토한다. 각 범주 50장, 총 150장이다. PowerShell에서 다음을
실행한다.

```powershell
& 'C:\Users\USER\Desktop\Capillary_limb\label\run_pa_150.cmd'
```

`run_pa_150.cmd`는 `section4_manual_review.py --independent`를 실행한다.
모든 영상의 U/D와 두 분지 측정선은 처음에 비어 있으며, 기존 분지 CSV와 MTL
학습 키포인트는 불러오지 않는다. 화면에는 원본 RGB ROI만 표시한다. 처음 실행 전
입력을 확인하려면 `run_pa_150.cmd --check`를 실행한다. 확인 모드는 CSV를 쓰지 않는다.

승인 시 `label/outputs/section4_review_pa_150/manual_review_150.csv`에 영상별
좌표와 세 직경(px)을 저장한다. 다시 실행하면 이 CSV의 작업을 이어서 불러온다.
`Filename`은 원본 파일명, `shape`는 범주다. 범주별 파일명이 중복되면 오류를
내므로 다른 혈관의 결과가 같은 CSV 행에 합쳐지지 않는다. 측정을 시작한 후
선정 영상을 바꿨다면 기존 CSV의 좌표가 새 영상에 잘못 붙지 않는지 확인해야 한다.

U/D는 루프 정점의 양쪽 경계점이고, B1/B2는 각 분지의 횡단 직경을 정하는
양끝점이다. 승인하면 `manual_loop_px`, `B1_px`, `B2_px`,
`manual_thin_px`, `manual_wide_px`가 채워진다. B1/B2는 혈류 방향으로 확인된
세동맥·세정맥 라벨이 아니다. 작은 직경과 큰 직경은 형태학적 추정으로만 해석한다.

## 기존 51장 검수

`section4_manual_review.py`는 선정된 51장의 **원본 RGB ROI**를 보여준다. 기존
`analysis/total/image/passivity.csv`의 두 분지 측정선은 **미승인 초안**으로
불러온다. 기본 모드는 MTL 학습 키포인트 CSV의 U/D도 초안으로 불러와 **검수**한다.
독립적인 제4절 루프 수동 측정을 원하면 `--blind-loop`로 시작한다. 이 경우
새 영상의 U/D는 비어 있다. 기존 영상·CSV는 수정하지 않는다. 자동 분할 마스크나
자동 직경은 화면에 표시하지 않는다.

## 측정 규칙

1. 각 영상에서 정점의 같은 단면에 있는 두 경계점을 U/D로 기록한다. 정점부에서
   보이는 혈류 기둥의 최대 국소 직경을 선택하고, 측정선은 국소 주행 방향에
   가능한 한 수직으로 놓는다. 기존 U/D도 원본 영상에서 확인한다.
2. 정점부를 제외한 **각각의 혈관 분지**에서 보이는 경계 사이 최대 국소 직경을
   수직 단면으로 기록한다. 혈관 겹침, 다른 혈관의 경계, ROI 잘림 부위를
   측정선으로 채택하지 않는다. 매 영상에서 같은 대상과 규칙을 적용한다.
3. `B1`, `B2`는 기존 CSV의 첫째·둘째 측정선일 뿐 세동맥·세정맥이나 화면의
   좌·우를 뜻하지 않는다. 꼬인 혈관에서는 분지의 물리적 연결을 확인한다.
   검수 뒤 작은 직경을 `manual_thin_px`, 큰 직경을 `manual_wide_px`로 계산한다.
   이 둘은 형태학적 **추정** 세동맥측·세정맥측에 대응한다.
4. 경계나 동일 분지를 신뢰성 있게 식별할 수 없으면 `unmeasurable`로 표시하고
   이유를 기록한다. 승인할 수 있는 영상만 `approved`로 표시한다.

## 실행

프로젝트 루트에서:

```powershell
$py = 'C:\Users\USER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py .\label\section4_manual_review.py --check
& $py .\label\section4_manual_review.py
# 독립적인 새 루프 측정: 처음부터 별도 결과 파일 사용
& $py .\label\section4_manual_review.py --blind-loop --output-csv '.\label\outputs\section4_review\manual_review_blind_loop.csv'
```

라디오 버튼으로 점 하나를 고르고 원본 영상에서 클릭해 수정한다. 화면 확대 후
스크롤할 수 있다. `Approve all three diameters`를 누르면 여섯 점이 모두 있고
각 선의 길이가 양수인지 검사한 뒤 승인한다. `Mark unmeasurable`에는 이유가
필수다. 화면 이동이나 종료 시 작업 내용은
`label/outputs/section4_review/manual_review.csv`에 저장되어 다음 실행에서
이어진다. 별도 `--output-csv`를 지정했으면 그 위치에 저장한다. 저장된 검수
파일은 `--blind-loop` 여부와 관계없이 **기존 점을 그대로 재개**하므로, 이미
U/D를 가져온 기본 모드와 독립 측정을 섞지 않으려면 처음부터 별도 결과 파일을
사용한다. `pending`에는 직경 결과 열을 비워 두므로 최종 평가에서 제외한다.

측정값은 512×512 ROI의 픽셀 단위다. 물리 단위(μm)로 환산할 때는 촬영 영상의
실제 μm/px 보정값이 필요하다. 논문 최종 비교에는 51개 영상 각각이 학습에서
제외됐던 외부 fold의 MTL 예측 결과를 사용한다.
