'''
검출못한 이미지 따로 저장
세동맥 세정맥 검출 코드
'''

import os
import cv2
import pandas as pd
import numpy as np
from analyzer import analyze_single_image

# ================= 설정 영역 =================
IMAGE_DIR = r"C:\Users\park_younguk\Desktop\analysis\effi_mask"
KEYPOINT_CSV = r"C:\Users\park_younguk\Desktop\analysis\efficientNet-b2.csv"

OUTPUT_CSV = os.path.join(IMAGE_DIR, "final_mtl_algo_measurement.csv")
VISUAL_DIR = os.path.join(IMAGE_DIR, "algo_visual_check")
# [추가] 실패한 이미지만 따로 모을 폴더
FAILED_DIR = os.path.join(IMAGE_DIR, "algo_failed_samples") 
# =============================================

def main():
    print(" 1,399장 대량 분석 및 데이터 정제 시작...")
    
    # 폴더 생성 (결과 확인용 & 실패 수집용)
    for folder in [VISUAL_DIR, FAILED_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"폴더 생성 완료: {folder}")

    if not os.path.exists(KEYPOINT_CSV):
        print(f"키포인트 CSV 파일이 없습니다: {KEYPOINT_CSV}")
        return
    
    df_raw = pd.read_csv(KEYPOINT_CSV)
    df_keypoints = df_raw.rename(columns={
        'pred_U_x': 'U_x', 'pred_U_y': 'U_y',
        'pred_D_x': 'D_x', 'pred_D_y': 'D_y',
        'filename': 'filename'
    })
    df_keypoints['pure_filename'] = df_keypoints['filename'].apply(lambda x: os.path.splitext(str(x))[0])

    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".tif"))])
    
    if not image_files:
        print(f"❌ 이미지 파일이 없습니다: {IMAGE_DIR}")
        return

    results_list = []

    print("-" * 60)
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        pure_img_name = os.path.splitext(img_name)[0]
        row_data = df_keypoints[df_keypoints['pure_filename'] == pure_img_name]

        row = {
            "Filename": img_name,
            "A_x1": None, "A_y1": None, "A_x2": None, "A_y2": None, "Algo_Arterial_Diameter(px)": None,
            "V_x1": None, "V_y1": None, "V_x2": None, "V_y2": None, "Algo_Venous_Diameter(px)": None,
            "Status": "Failed"
        }

        if not row_data.empty:
            result = analyze_single_image(img_path, df_keypoints)
            
            # 시각화를 위한 이미지 로드 (한글 경로 대응)
            img_array = np.fromfile(img_path, np.uint8)
            img_vis = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # 성공 조건: 분석 결과가 OK이고 혈관이 최소 2개 검출되었을 때
            if result["ok"] and len(result["leg_results"]) >= 2:
                leg_results = result["leg_results"]
                
                # [로직 변경] 직경(max_d) 기준으로 정렬 (작은 것: 세동맥, 큰 것: 세정맥)
                sorted_legs_by_size = sorted(leg_results, key=lambda x: x["max_d"])
                
                arterial_leg = sorted_legs_by_size[0] # 더 가는 혈관
                venous_leg = sorted_legs_by_size[1]   # 더 굵은 혈관\

                # --- 1. 세동맥 (빨간색) 계산 및 그리기 ---
                ax1, ay1 = int(round(arterial_leg["max_x"] + arterial_leg["local_perp_v"][0] * arterial_leg["max_d_pos"])), \
                           int(round(arterial_leg["max_y"] + arterial_leg["local_perp_v"][1] * arterial_leg["max_d_pos"]))
                ax2, ay2 = int(round(arterial_leg["max_x"] - arterial_leg["local_perp_v"][0] * arterial_leg["max_d_neg"])), \
                           int(round(arterial_leg["max_y"] - arterial_leg["local_perp_v"][1] * arterial_leg["max_d_neg"]))
                
                row.update({"A_x1": ax1, "A_y1": ay1, "A_x2": ax2, "A_y2": ay2, "Algo_Arterial_Diameter(px)": round(arterial_leg["max_d"], 2)})
                cv2.line(img_vis, (ax1, ay1), (ax2, ay2), (0, 0, 255), 2)
                cv2.putText(img_vis, f"A: {arterial_leg['max_d']:.1f}px", (ax1, ay1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # --- 2. 세정맥 (파란색) 계산 및 그리기 ---
                vx1, vy1 = int(round(venous_leg["max_x"] + venous_leg["local_perp_v"][0] * venous_leg["max_d_pos"])), \
                           int(round(venous_leg["max_y"] + venous_leg["local_perp_v"][1] * venous_leg["max_d_pos"]))
                vx2, vy2 = int(round(venous_leg["max_x"] - venous_leg["local_perp_v"][0] * venous_leg["max_d_neg"])), \
                           int(round(venous_leg["max_y"] - venous_leg["local_perp_v"][1] * venous_leg["max_d_neg"]))
                ''
                row.update({"V_x1": vx1, "V_y1": vy1, "V_x2": vx2, "V_y2": vy2, "Algo_Venous_Diameter(px)": round(venous_leg["max_d"], 2)})
                cv2.line(img_vis, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                cv2.putText(img_vis, f"V: {venous_leg['max_d']:.1f}px", (vx1, vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                row["Status"] = "Success"
                print(f"[{idx+1}/{len(image_files)}] {img_name} 분석 완료")

                # 시각화 이미지 저장 (성공)
                _, img_encoded = cv2.imencode(".png", img_vis)
                img_encoded.tofile(os.path.join(VISUAL_DIR, f"check_{img_name}"))
            
            else:
                # [추가] 실패한 경우 원인 표시 및 전용 폴더 저장
                detected_count = len(result.get("leg_results", []))
                print(f"[{idx+1}/{len(image_files)}] ❌ {img_name} 측정 실패 (혈관 {detected_count}개 검출)")
                
                cv2.putText(img_vis, "ALGO_FAILED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img_vis, f"Detected Legs: {detected_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                _, img_encoded = cv2.imencode(".png", img_vis)
                img_encoded.tofile(os.path.join(FAILED_DIR, f"fail_{img_name}"))
        
        else:
            print(f"[{idx+1}/{len(image_files)}] {img_name} 건너뜀 (CSV 데이터 없음)")

        results_list.append(row)

    # 최종 CSV 저장
    pd.DataFrame(results_list).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("-" * 60)
    print(f"모든 작업이 완료되었습니다!")
    print(f"결과 파일: {OUTPUT_CSV}")
    print(f"성공 시각화: {VISUAL_DIR}")
    print(f"실패 샘플 모음: {FAILED_DIR}")

if __name__ == "__main__":
    main()