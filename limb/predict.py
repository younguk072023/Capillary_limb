import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. 폴더 구조에 맞게 경로 설정 (limb 폴더에서 실행 시 부모 폴더를 path에 추가)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 2. 상위 label 폴더 안의 생성기 3개 임포트
from label.U_label import generate_realistic_normal_u
from label.Tortuous import generate_clinical_tortuous_mask
from label.Crossing_label import generate_clinical_crossing_mask

# 3. 측정 알고리즘 임포트
from analyzer import analyze_single_image

def run_accuracy_test(samples_per_type=30):
    results = []
    temp_dir = os.path.join(current_dir, "temp_val_images")
    viz_dir = os.path.join(current_dir, "viz_results") # 🌟 시각화 이미지가 저장될 폴더
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # 3가지 형태별 생성 함수 매핑
    categories = [
        ("Normal", generate_realistic_normal_u),
        ("Tortuous", generate_clinical_tortuous_mask),
        ("Crossing", generate_clinical_crossing_mask)
    ]

    print(f"🚀 [계층적 프레임워크 정확도 검증 & 시각화] 총 {len(categories) * samples_per_type}개 분석 시작...\n")

    for cat_name, gen_func in categories:
        print(f"📦 [{cat_name}] 형태 {samples_per_type}장 생성 및 분석 중...")

        # 진행률 표시 바 적용
        for i in tqdm(range(samples_per_type), desc=cat_name):
            # 무작위 두께 및 기본 좌표 설정
            rand_art = random.randint(12, 18)
            rand_ven = int(rand_art * random.uniform(1.2, 1.7))
            cx_val, top_y_val, bottom_y_val = 256, 100, 480

            # 반환값이 파일마다 조금씩 다르므로 분기 처리
            if cat_name == "Normal":
                is_perf = random.choice([True, False])
                img_rgb, _, gt_info = gen_func(cx=cx_val, top_y=top_y_val, bottom_y=bottom_y_val, art_thick=rand_art, ven_thick=rand_ven, is_perfect=is_perf)
                gt_art, gt_ven = gt_info

            elif cat_name == "Tortuous":
                img_rgb, _, _, gt_info = gen_func(cx=cx_val, top_y=top_y_val, bottom_y=bottom_y_val, art_thick=rand_art, ven_thick=rand_ven)
                gt_art, gt_ven = gt_info['art_max_thick'], gt_info['ven_max_thick']

            elif cat_name == "Crossing":
                img_rgb, _, gt_info, _ = gen_func(cx=cx_val, top_y=top_y_val, bottom_y=bottom_y_val, art_thick=rand_art, ven_thick=rand_ven)
                gt_art, gt_ven = gt_info

            # 🌟 [핵심] 알고리즘 분석용 순수 흰색 마스크 추출 (선 제거)
            clean_mask = np.all(img_rgb == [255, 255, 255], axis=-1).astype(np.uint8) * 255
            temp_img_path = os.path.join(temp_dir, f"{cat_name}_{i}.png")
            cv2.imwrite(temp_img_path, clean_mask)

            # 분석 엔진이 사용할 가상의 U-point, D-point 좌표 DataFrame
            fake_df = pd.DataFrame([{
                "filename": f"{cat_name}_{i}",
                "U_x": cx_val, "U_y": top_y_val,
                "D_x": cx_val, "D_y": bottom_y_val - 150
            }])

            # 알고리즘 분석 수행
            try:
                res = analyze_single_image(temp_img_path, fake_df)
                if res["ok"] and len(res["leg_results"]) >= 2:
                    # 두께순으로 정렬되므로 얇은 쪽을 A, 굵은 쪽을 V로 매칭
                    legs = res["leg_results"]
                    p1, p2 = legs[0]["max_d"], legs[1]["max_d"]
                    pred_art = min(p1, p2)
                    pred_ven = max(p1, p2)

                    err_a = abs(gt_art - pred_art)
                    err_v = abs(gt_ven - pred_ven)

                    results.append({
                        "Type": cat_name,
                        "GT_A": gt_art, "Pred_A": round(pred_art, 2), "Err_A": round(err_a, 2),
                        "GT_V": gt_ven, "Pred_V": round(pred_ven, 2), "Err_V": round(err_v, 2)
                    })

                    # =========================================================
                    # 🌟 [시각화 이미지 저장 로직]
                    # 원본 img_rgb(빨강/파랑 선 그려진 정답 이미지) 위에 텍스트 오버레이
                    # =========================================================
                    viz_img = img_rgb.copy()
                    
                    # 텍스트 가독성을 높이기 위한 반투명 검정색 배경 박스
                    overlay = viz_img.copy()
                    cv2.rectangle(overlay, (10, 10), (450, 90), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, viz_img, 0.4, 0, viz_img)
                    
                    # 정답(GT)과 알고리즘 예측값(Pred) 텍스트 쓰기
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # 세동맥 (빨간선 계열)
                    cv2.putText(viz_img, f"[Art] GT:{gt_art} px | Pred:{pred_art:.1f} px", (20, 40), font, 0.7, (255, 100, 100), 2)
                    # 세정맥 (파란선 계열)
                    cv2.putText(viz_img, f"[Ven] GT:{gt_ven} px | Pred:{pred_ven:.1f} px", (20, 75), font, 0.7, (100, 100, 255), 2)
                    
                    # 오차가 3px 이상이면 눈에 띄게 경고색 표시 (디버깅용)
                    if err_a >= 3 or err_v >= 3:
                        cv2.rectangle(viz_img, (0, 0), (511, 511), (0, 165, 255), 4) # 주황색 테두리
                        
                    # 시각화 파일 저장 (BGR 형태로 변환하여 저장)
                    viz_path = os.path.join(viz_dir, f"{cat_name}_{i:02d}.png")
                    cv2.imwrite(viz_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))

            except Exception as e:
                pass # 에러가 난 샘플은 통과

            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    # 4. 분석 결과 데이터프레임 요약 (기존과 동일)
    df = pd.DataFrame(results)
    if not df.empty:
        summary = df.groupby("Type").agg(
            Count=("Type", "count"),
            MAE_A=("Err_A", "mean"),
            MAE_V=("Err_V", "mean")
        ).reset_index()

        summary["MAE_A"] = summary["MAE_A"].round(2)
        summary["MAE_V"] = summary["MAE_V"].round(2)

        print("\n" + "="*55)
        print("🏆 [계층적 프레임워크 - 모세혈관 직경 측정 정확도 리포트]")
        print("="*55)
        print(summary.to_string(index=False))
        print("-" * 55)
        
        total_mae_a = df["Err_A"].mean()
        total_mae_v = df["Err_V"].mean()
        print(f"📍 전체 세동맥(A) 평균 절대 오차(MAE): {total_mae_a:.2f} px")
        print(f"📍 전체 세정맥(V) 평균 절대 오차(MAE): {total_mae_v:.2f} px")
        print(f"🔥 시스템 통합 측정 오차: {(total_mae_a + total_mae_v)/2:.2f} px")
        print(f"📁 시각화 결과 저장 완료: {viz_dir} 폴더를 확인하세요!")
        print("="*55)
    else:
        print("⚠️ 분석된 결과가 없습니다.")

    # 임시 폴더 삭제 (temp만 지우고 viz_results는 남김)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

if __name__ == "__main__":
    # 데이터 재현성(항상 같은 결과)을 위한 시드 고정
    random.seed(42)
    np.random.seed(42)
    run_accuracy_test()