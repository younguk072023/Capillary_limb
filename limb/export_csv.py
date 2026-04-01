import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# 영욱님의 analyzer 모듈 임포트
from analyzer import analyze_single_image

def save_viz_image_for_qc(image_name, result, output_folder):
    """
    CapillaryViewer의 스타일을 적용하여, 
    L/R 경로, 비대칭 측정선, 수치를 그려서 저장합니다.
    """
    # 다크모드 스타일 적용
    plt.style.use('dark_background')
    
    # 해상도를 높여서 저장하기 위해 DPI 설정
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # 데이터 추출
    img = result["img"]
    leg_results = result["leg_results"] # Analyzer에서 x좌표순(L->R)으로 정렬되어 있음
    labeled_mask = result["labeled_mask"]
    U_xy, D_xy = result["U_xy"], result["D_xy"]
    used_branch_pt = result["used_branch_pt"]
    dir_v = result["dir_v"]
    apex_cut_pt = result["apex_cut_pt"]

    # 1. 원본 이미지 표시
    ax.imshow(img, cmap="gray")
    
    # 2. 파티션 마스크 오버레이 (L/R 영역 분할 영역)
    overlay = np.ma.masked_where(labeled_mask == 0, labeled_mask)
    ax.imshow(overlay, cmap="nipy_spectral", alpha=0.20)

    # 제목 설정
    sorted_for_title = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)
    title_text = f"QC Viz: {os.path.basename(image_name)}\n"
    if len(sorted_for_title) >= 2:
        title_text += f"V: {sorted_for_title[0]['max_d']:.2f}px | A: {sorted_for_title[1]['max_d']:.2f}px"
    ax.set_title(title_text, color='white', fontsize=18, fontweight='bold', pad=15)

    # -------------------------------------------------------------
    # 3. 각 다리(L/R)별 경로 및 측정선 그리기
    # -------------------------------------------------------------
    for idx, res in enumerate(leg_results):
        # Viewer와 색상 통일: 0은 Left(Cyan), 1은 Right(Yellow)
        side_color = "cyan" if idx == 0 else "yellow"
        
        # 3-1. 측정 경로(Path) 그리기
        tx = [p[1] for p in res["trimmed_path"]]
        ty = [p[0] for p in res["trimmed_path"]]
        # 테두리
        ax.plot(tx, ty, color="black", linewidth=5, zorder=2)
        # 실선
        ax.plot(tx, ty, color=side_color, linewidth=2.5, zorder=3)

        # 3-2. 비대칭 양방향 측정선 그리기
        mx, my = res["max_x"], res["max_y"]
        pv = res["local_perp_v"]
        d_pos, d_neg = res["max_d_pos"], res["max_d_neg"]
        
        # 양끝점 계산
        p1 = (mx + pv[0] * d_pos, my + pv[1] * d_pos)
        p2 = (mx - pv[0] * d_neg, my - pv[1] * d_neg)
        
        # 측정선 테두리
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=6, zorder=5)
        # 측정 실선 (빨간색)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", linewidth=3, zorder=6)
        
        # 3-3. 타점 및 수치 표시
        # 혈관 라벨 매칭 (동맥/정맥)
        label_sort = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)
        is_venous = (res["id"] == label_sort[0]["id"])
        final_label = "Ven" if is_venous else "Art"
        
        # 수치 텍스트 표시
        text_y_offset = (d_pos + d_neg) / 2.0 + 15
        ax.text(mx, my - text_y_offset, 
                f"{final_label}: {res['max_d']:.2f}", 
                color="white", fontsize=11, fontweight='bold', ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec=side_color, alpha=0.8, linewidth=1.5), zorder=8)

    # -------------------------------------------------------------
    # 4. 주요 기준점 표시 (디버깅용)
    # -------------------------------------------------------------
    perp_v = np.array([-dir_v[1], dir_v[0]])
    D_vec = np.array(D_xy)
    
    # Apex 폭 측정 기준선 (Red)
    ax.plot([D_vec[0]+perp_v[0]*150, D_vec[0]-perp_v[0]*150], [D_vec[1]+perp_v[1]*150, D_vec[1]-perp_v[1]*150], color="red", ls="-", lw=2, alpha=0.7)
    
    # 바닥 자르기 기준선 (Blue) - 있는 경우만
    if used_branch_pt is not None:
        B_vec = np.array([used_branch_pt[1], used_branch_pt[0]])
        ax.plot([B_vec[0]+perp_v[0]*150, B_vec[0]-perp_v[0]*150], [B_vec[1]+perp_v[1]*150, B_vec[1]-perp_v[1]*150], color="blue", ls="-", lw=2, alpha=0.7)
        ax.scatter(B_vec[0], B_vec[1], c="lime", s=80, edgecolors="black", marker="s", zorder=7)

    # U/D 포인트, Apex 절단점
    ax.scatter(U_xy[0], U_xy[1], c="gray", s=60, edgecolors="white", zorder=7)
    ax.scatter(D_xy[0], D_xy[1], c="red", s=70, edgecolors="white", zorder=7)
    if apex_cut_pt is not None:
        ax.scatter(apex_cut_pt[1], apex_cut_pt[0], c="magenta", s=90, edgecolors="white", marker="x", zorder=7)

    # 5. 최종 저장 및 플롯 닫기
    ax.axis("off")
    fig.tight_layout()
    
    # 이미지 이름과 동일하게 png로 저장
    save_filename = os.path.splitext(image_name)[0] + ".png"
    plt.savefig(os.path.join(output_folder, save_filename), bbox_inches="tight", pad_inches=0.1)
    
    plt.close(fig)
    # 스타일 초기화 (다른 플롯에 영향 주지 않기 위해)
    plt.style.use('default')

# =========================================================
# 5. 메인 추출 함수 (CSV + Viz 통합)
# =========================================================
def export_csv_and_viz_for_qc(image_dir, csv_path, output_csv_name="capillary_measurements_for_qc.csv"):
    df_keypoints = pd.read_csv(csv_path)
    
    valid_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])
    
    if len(image_files) == 0:
        print(f"❌ 이미지 파일이 없습니다: {image_dir}")
        return

    # 시각화 저장 폴더 생성
    viz_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz_for_qc")
    os.makedirs(viz_out_dir, exist_ok=True)

    print(f"🚀 총 {len(image_files)}개 이미지 분석 시작")
    print(f"📁 [CSV] {output_csv_name}")
    print(f"📁 [QC 이미지] viz_for_qc/ 폴더")
    print("="*60 + "\n")
    
    results_list = []
    success_count = 0

    # 🌟 메인 루프 (tqdm 진행바)
    for img_name in tqdm(image_files, desc="Exporting Data & Viz"):
        img_path = os.path.join(image_dir, img_name)
        
        # 1. 단일 이미지 분석
        try:
            result = analyze_single_image(img_path, df_keypoints)
        except Exception as e:
            print(f"\n[ERR] {img_name} 분석 중 치명적 오류: {e}")
            continue
        
        # 분석 실패 또는 다리가 2개 미만인 경우 패스
        if not result["ok"] or len(result["leg_results"]) < 2:
            continue
            
        leg_results = result["leg_results"]
        
        # -------------------------------------------------------------
        # 🌟 [핵심] 육안 검증용 시각화 이미지 저장 호출
        # -------------------------------------------------------------
        try:
            save_viz_image_for_qc(img_name, result, viz_out_dir)
        except Exception as e:
            print(f"\n[WARN] {img_name} 시각화 저장 중 오류 (데이터는 추출됨): {e}")

        # -------------------------------------------------------------
        # 2. CSV 데이터 추출 로직 (이전과 동일)
        # -------------------------------------------------------------
        # 두께순 정렬 (V > A)
        sorted_legs = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)
        ven_res = sorted_legs[0]
        art_res = sorted_legs[1]

        # 양끝점 좌표 계산 및 좌(L)/우(R) 정렬 함수
        def get_lr_coordinates(res):
            mx, my = res["max_x"], res["max_y"]
            pv = res["local_perp_v"]
            d_pos = res["max_d_pos"]
            d_neg = res["max_d_neg"]
            
            p1_x, p1_y = mx + pv[0] * d_pos, my + pv[1] * d_pos
            p2_x, p2_y = mx - pv[0] * d_neg, my - pv[1] * d_neg
            
            if p1_x < p2_x:
                return round(p1_x, 2), round(p1_y, 2), round(p2_x, 2), round(p2_y, 2)
            else:
                return round(p2_x, 2), round(p2_y, 2), round(p1_x, 2), round(p1_y, 2)

        a_l_x, a_l_y, a_r_x, a_r_y = get_lr_coordinates(art_res)
        v_l_x, v_l_y, v_r_x, v_r_y = get_lr_coordinates(ven_res)

        # 결과 리스트에 추가
        results_list.append({
            "Filename": img_name,
            "A_Diameter": round(art_res["max_d"], 2),
            "A_L_x": a_l_x, "A_L_y": a_l_y,
            "A_R_x": a_r_x, "A_R_y": a_r_y,
            "V_Diameter": round(ven_res["max_d"], 2),
            "V_L_x": v_l_x, "V_L_y": v_l_y,
            "V_R_x": v_r_x, "V_R_y": v_r_y
        })
        success_count += 1

    # 3. CSV 저장
    df_out = pd.DataFrame(results_list)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_csv_name)
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*60)
    print(f"✅ 분석 완료! 총 {len(image_files)}개 중 {success_count}개 성공")
    print(f"📄 CSV 결과: {save_path}")
    print(f"📁 QC 이미지: {viz_out_dir}/ (반드시 육안으로 확인하세요!)")
    print("="*60)

if __name__ == "__main__":
    # 영욱 님의 실제 데이터 경로 설정
    CSV_PATH = r"capillary_keypoint_final.csv"
    LABEL_DIR = r"D:\usb\MTL_dataset\label"
    
    export_csv_and_viz_for_qc(LABEL_DIR, CSV_PATH)