'''
이상치 분류 IsolationForest
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. 경로 설정 (선생님의 환경에 맞게 수정하세요!)
# ==========================================
csv_file_path = 'limb_total.csv'        # 정량화 데이터 CSV 파일
source_img_folder = r'E:\MTL_dataset\image'                # 원본 크롭 이미지들이 들어있는 폴더 이름
output_anomaly_folder = './detected_anomalies' # 이상치 이미지들을 모아둘 새 폴더 이름 (자동 생성됨)

# ==========================================
# 2. 데이터 불러오기 및 머신러닝 분석
# ==========================================
df = pd.read_csv(csv_file_path)

# 분석에 사용할 3가지 형태적 특성(Feature)
features = ['loop_length', 'arterial_diameter', 'venous_diameter']
X = df[features]

# 데이터 스케일링 (단위 통일)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 고립 숲(Isolation Forest) 모델 학습 (상위 5%를 이상치로 간주)
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(X_scaled)
df['anomaly_score'] = model.decision_function(X_scaled)

# ==========================================
# 3. 이상치(Outlier) 분류 및 결과 정리
# ==========================================
normal_df = df[df['anomaly'] == 1]
outlier_df = df[df['anomaly'] == -1].sort_values(by='anomaly_score')

print("=== 📊 분석 결과 요약 ===")
print(f"총 분석 데이터: {len(df)}개")
print(f"정상 패턴 군집: {len(normal_df)}개")
print(f"형태적 이상치(비율 붕괴): {len(outlier_df)}개 발견!\n")

# ==========================================
# 4. 이상치 이미지 자동 추출 및 폴더 저장
# ==========================================
# 새 폴더 만들기 (이미 있으면 덮어쓰지 않고 그대로 사용)
os.makedirs(output_anomaly_folder, exist_ok=True)

print(f"=== 📁 이상치 이미지 복사 시작 (저장 폴더: {output_anomaly_folder}/) ===")

copied_count = 0
for index, row in outlier_df.iterrows():
    filename = row['filename']
    score = row['anomaly_score']
    
    # 원본 파일 경로와 복사될 새 경로
    src_path = os.path.join(source_img_folder, filename)
    dst_path = os.path.join(output_anomaly_folder, filename)
    
    # 파일이 실제로 존재하는지 확인 후 복사
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied_count += 1
        print(f" [복사 완료] {filename} (이상치 점수: {score:.3f})")
    else:
        print(f" [경고] {filename} 파일을 {source_img_folder} 폴더에서 찾을 수 없습니다.")

print(f"\n✅ 총 {copied_count}개의 이상치 이미지가 성공적으로 분류/저장되었습니다.")

# 연구 분석용으로 이상치들만의 스펙을 담은 요약 CSV도 해당 폴더에 하나 만들어 줍니다.
summary_csv_path = os.path.join(output_anomaly_folder, 'anomaly_summary.csv')
outlier_df[['filename', 'loop_length', 'arterial_diameter', 'venous_diameter', 'anomaly_score']].to_csv(summary_csv_path, index=False)
print(f"✅ 이상치 요약 데이터({summary_csv_path}) 저장 완료.")

# ==========================================
# 5. 3D 시각화 (선택 사항)
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(normal_df['arterial_diameter'], normal_df['venous_diameter'], normal_df['loop_length'], 
           c='blue', label='Normal (Inlier)', alpha=0.3)
ax.scatter(outlier_df['arterial_diameter'], outlier_df['venous_diameter'], outlier_df['loop_length'], 
           c='red', label='Anomaly (Outlier)', s=100, edgecolor='k')

ax.set_xlabel('Arterial Diameter')
ax.set_ylabel('Venous Diameter')
ax.set_zlabel('Loop Diameter')
ax.set_title('Multivariate Anomaly Detection')
ax.legend()
plt.show()