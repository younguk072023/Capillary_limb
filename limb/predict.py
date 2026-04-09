'''
알고리즘 분석 코드
'''
import os
import cv2
import pandas as pd
import numpy as np
from analyzer import analyze_single_image

# ================= 설정 영역 =================
IMAGE_DIR = r"C:\Users\park_younguk\Desktop\analysis\total\label"  
KEYPOINT_CSV = r"capillary_keypoint_final.csv"          
OUTPUT_CSV = os.path.join(IMAGE_DIR, "algo_gt_measurement_unified.csv")
VISUAL_DIR = os.path.join(IMAGE_DIR, "algo_visual_check")
# =============================================

def main():
    print("[좌우 고정 모드] 알고리즘 자동 측정을 시작합니다...")
    
    if not os.path.exists(VISUAL_DIR):
        os.makedirs(VISUAL_DIR)
        print(f"시각화 폴더 생성 완료: {VISUAL_DIR}")

    if not os.path.exists(KEYPOINT_CSV):
        print(f"키포인트 CSV 파일이 없습니다: {KEYPOINT_CSV}")
        return
    df_keypoints = pd.read_csv(KEYPOINT_CSV)

    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".tif"))])
    
    if not image_files:
        print(f"이미지 파일이 없습니다: {IMAGE_DIR}")
        return

    results_list = []

    print("-" * 50)
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        result = analyze_single_image(img_path, df_keypoints)
        
        img_array = np.fromfile(img_path, np.uint8)
        img_vis = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        row = {
            "Filename": img_name,
            "A_x1": None, "A_y1": None, "A_x2": None, "A_y2": None, "Algo_Arterial_Diameter(px)": None,
            "V_x1": None, "V_y1": None, "V_x2": None, "V_y2": None, "Algo_Venous_Diameter(px)": None,
            "Status": "Failed"
        }

        if result["ok"] and len(result["leg_results"]) >= 2:
            leg_results = result["leg_results"]
            
            # [핵심 수정] 두께가 아니라 가로 좌표(max_x) 기준으로 정렬
            # x값이 작은 쪽이 0번(왼쪽), 큰 쪽이 1번(오른쪽)
            sorted_legs = sorted(leg_results, key=lambda x: x["max_x"])
            
            left_leg = sorted_legs[0]   # 왼쪽 (Arterial 칸에 저장)
            right_leg = sorted_legs[1]  # 오른쪽 (Venous 칸에 저장)

            # 1. 왼쪽 혈관 (A컬럼 저장 / 빨간색 표시)
            ax1, ay1 = int(round(left_leg["max_x"] + left_leg["local_perp_v"][0] * left_leg["max_d_pos"])), int(round(left_leg["max_y"] + left_leg["local_perp_v"][1] * left_leg["max_d_pos"]))
            ax2, ay2 = int(round(left_leg["max_x"] - left_leg["local_perp_v"][0] * left_leg["max_d_neg"])), int(round(left_leg["max_y"] - left_leg["local_perp_v"][1] * left_leg["max_d_neg"]))
            row.update({"A_x1": ax1, "A_y1": ay1, "A_x2": ax2, "A_y2": ay2, "Algo_Arterial_Diameter(px)": round(left_leg["max_d"], 2)})
            
            cv2.line(img_vis, (ax1, ay1), (ax2, ay2), (0, 0, 255), 2) # 빨간색
            cv2.putText(img_vis, f"Left: {left_leg['max_d']:.1f}px", (ax1, ay1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 2. 오른쪽 혈관 (V컬럼 저장 / 파란색 표시)
            vx1, vy1 = int(round(right_leg["max_x"] + right_leg["local_perp_v"][0] * right_leg["max_d_pos"])), int(round(right_leg["max_y"] + right_leg["local_perp_v"][1] * right_leg["max_d_pos"]))
            vx2, vy2 = int(round(right_leg["max_x"] - right_leg["local_perp_v"][0] * right_leg["max_d_neg"])), int(round(right_leg["max_y"] - right_leg["local_perp_v"][1] * right_leg["max_d_neg"]))
            row.update({"V_x1": vx1, "V_y1": vy1, "V_x2": vx2, "V_y2": vy2, "Algo_Venous_Diameter(px)": round(right_leg["max_d"], 2)})
            
            cv2.line(img_vis, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2) # 파란색
            cv2.putText(img_vis, f"Right: {right_leg['max_d']:.1f}px", (vx1, vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            row["Status"] = "Success"
            print(f"[{idx+1}/{len(image_files)}] {img_name} 좌우 분석 완료")
        else:
            cv2.putText(img_vis, "FAILED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"[{idx+1}/{len(image_files)}]  {img_name} 실패")

        # 시각화 이미지 저장
        result_name = f"check_{img_name}"
        _, img_encoded = cv2.imencode(".png", img_vis)
        img_encoded.tofile(os.path.join(VISUAL_DIR, result_name))
        
        results_list.append(row)

    pd.DataFrame(results_list).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("-" * 50)
    print(f" 좌우 기준 통합 완료! 시각화 확인: {VISUAL_DIR}")

if __name__ == "__main__":
    main()