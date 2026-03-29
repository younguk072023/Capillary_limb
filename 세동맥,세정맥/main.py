'''
메인 실행 파일
1  이미지 읽은 Skeleton
2. CSV에서 해당 이미지의 U,D 좌표 읽기
3. U와 D의 중간점을 Apex 시작점으로 잡기\
4. branch 후보 찾기
5. skeleton을 두다리로 분리
6. 각 다리의 유효 구간을 Curvature기반으로 잘라냄.
7. 그 구간에서 최대 직경을 계산한.
8. 결과 정령 후 시각화

'''
import cv2
import numpy as np
import pandas as pd
import os

from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from image_utils import get_branch_candidate, split_two_legs_with_retry
from geometry import order_component_path, find_limb_valid_range_by_curvature
from visualize import plot_and_print_results

def skeletonize_image(image_path, csv_file):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return

    binary_image = img > 127
    skeleton = skeletonize(binary_image)
    labeled_full = label(skeleton)  #서로연결되어있는 것을 번호로 라벨 처리
    skeleton = (labeled_full == np.argmax(np.bincount(labeled_full.flat)[1:]) + 1)  #2차원 이미지 행렬을 1차원 한 줄로 길게 펼치는 것을 말함. 배경 0은 무시

    # CSV 데이터 검출 및 U, D 
    df = pd.read_csv(csv_file)
    search_name = os.path.splitext(os.path.basename(image_path))[0]
    row = df[df["filename"].apply(lambda x: os.path.splitext(str(x))[0]) == search_name]
    if row.empty: 
        return

    ux, uy = row["U_x"].values[0], row["U_y"].values[0]
    dx, dy = row["D_x"].values[0], row["D_y"].values[0]

    ys, xs = np.where(skeleton > 0) #흰색 배경의 이진화 이미지에서 skeletonize만 찾기위해
    skeleton_points = np.column_stack((xs, ys)) #가로, 세로 세트로 묶음
    mid_x, mid_y = (ux + dx) / 2, (uy + dy) / 2 #u와 d중간지점 스켈레톤을 시작점으로 지정 

    # Apex 및 Branch 추출
    smx, smy = skeleton_points[np.argmin(np.linalg.norm(skeleton_points - np.array([mid_x, mid_y]), axis=1))]
    apex_pt = (int(smy), int(smx))
    branch_pt, branch_xs, branch_ys = get_branch_candidate(skeleton, apex_pt)

    # 해부학적 분리 (다리 분할)
    split_result = split_two_legs_with_retry(skeleton, apex_pt, branch_pt)
    labeled_skel = split_result["labeled_skel"]
    selected_ids = split_result["selected_ids"]
    used_branch_pt = split_result["branch_pt"]

    # 거리 변환 (두께 맵 생성)
    _, indices = distance_transform_edt(labeled_skel == 0, return_indices=True)
    labeled_mask = labeled_skel[indices[0], indices[1]]
    labeled_mask[~binary_image] = 0
    dist_map = distance_transform_edt(binary_image)

    leg_results = []

    # 각 세동맥 세정맥 분석
    for comp_id in selected_ids:
        path = order_component_path((labeled_skel == comp_id), apex_ref=apex_pt, branch_ref=used_branch_pt)
        if len(path) < 2: continue

        # 곡률 기반 직선 구간 한정
        trimmed_path, start_pt, end_pt = find_limb_valid_range_by_curvature(path)
        if len(trimmed_path) == 0: continue

        trimmed_ys = np.array([p[0] for p in trimmed_path])
        trimmed_xs = np.array([p[1] for p in trimmed_path])

        # 최대 직경 산출
        bone_radii = dist_map[trimmed_ys, trimmed_xs]
        max_r_idx = np.argmax(bone_radii)

        leg_results.append({
            "id": comp_id,
            "max_d": float(bone_radii[max_r_idx] * 2),
            "max_x": int(trimmed_xs[max_r_idx]),
            "max_y": int(trimmed_ys[max_r_idx]),
            "mean_x": float(np.mean(trimmed_xs)),
            "path": path,
            "trimmed_path": trimmed_path,
            "start_pt": start_pt,
            "end_pt": end_pt
        })

    leg_results.sort(key=lambda x: x["mean_x"])

    # 시각화 호출
    plot_and_print_results(search_name, leg_results, labeled_mask, apex_pt, used_branch_pt)


if __name__ == "__main__":
    CSV_FILE = "capillary_keypoint_final.csv"
    skeletonize_image("p1_det_063.tif", CSV_FILE)