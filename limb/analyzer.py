import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import skeletonize
from utils import read_image_unicode, snap_xy_to_skeleton, smooth_1d 
from core_algorithm import (
    find_bottom_of_inner_hole,
    extract_two_leg_paths,
    trim_path_for_measurement,
    build_vessel_masks_from_side_skeletons
)

'''
한 개의 limb path가 주어졌을 때,
그 path를 따라가며 각 지점의 두께를 재고
그중에서 가장 안정적인 최대 직경을 뽑는 함수
'''

def get_stable_max_diameter_raycast(trimmed_path, binary_image, dir_v, D_xy,
                                    smooth_k=5, top_ratio=0, min_plateau_len=0):
    if len(trimmed_path) == 0:
        return None

    ys = np.array([p[0] for p in trimmed_path])
    xs = np.array([p[1] for p in trimmed_path])
    h, w = binary_image.shape
    
    D_vec = np.array([D_xy[0], D_xy[1]], dtype=float)  

    widths_raw = np.zeros(len(trimmed_path))    #각 지점의 두께를 적는 배열
    pvs = []    #각 지점에서 자를 어느 각도로 자를지 저장하는 배열 
    d_pos_list = [] #양쪽 거리
    d_neg_list = [] 



    for i in range(len(trimmed_path)):  # 각 지점마다 for문 돌려 길이 확인
        
        t_start = max(0, i - 3)
        t_end = min(len(trimmed_path) - 1, i + 3)
        
        dx_loc = xs[t_end] - xs[t_start]
        dy_loc = ys[t_end] - ys[t_start]
        norm_loc = np.hypot(dx_loc, dy_loc)
        
        #pv는 직경을 젤 방향
        if norm_loc < 1e-6:
            pv = np.array([-dir_v[1], dir_v[0]], dtype=float)
        else:
            pv = np.array([-dy_loc, dx_loc], dtype=float) / norm_loc

        
        # 양의 방향으로 limb 직경 측정
        d_pos = 0.0
        for step in range(1, 150):
            ny = int(round(ys[i] + pv[1] * step))
            nx = int(round(xs[i] + pv[0] * step))

            proj_D_ray = np.dot(np.array([nx, ny], dtype=float) - D_vec, dir_v)

            if (
                ny < 0 or ny >= h or nx < 0 or nx >= w
                or proj_D_ray < 0
                or not binary_image[ny, nx]
            ):
                d_pos = np.hypot(nx - xs[i], ny - ys[i])
                break
        
        # 음의 방향으로 limb 직경 측정
        d_neg = 0.0
        for step in range(1, 150):
            ny = int(round(ys[i] - pv[1] * step))
            nx = int(round(xs[i] - pv[0] * step))

            proj_D_ray = np.dot(np.array([nx, ny], dtype=float) - D_vec, dir_v)

            if (
                ny < 0 or ny >= h or nx < 0 or nx >= w
                or proj_D_ray < 0
                or not binary_image[ny, nx]
            ):
                d_neg = np.hypot(nx - xs[i], ny - ys[i])
                break
        
        # 양방향 거리의 합이 실제 단면적
        widths_raw[i] = d_pos + d_neg
        pvs.append(pv)
        d_pos_list.append(d_pos)
        d_neg_list.append(d_neg)

    # 노이즈를 방지하기 위해 15% 이상의 값들을 후보로 삼고, 그 중에서 가장 긴 구간이면서 평균이 큰 구간을 찾는 방식
    widths_s = smooth_1d(widths_raw, k=smooth_k)
    thr = np.quantile(widths_s, 1.0 - top_ratio)
    candidate_idx = np.where(widths_s >= thr)[0]

    # 모든 지점에서 소수점 단위까지 똑같을 경우. 물론 그런 일은 거의 없겠지만 혹싀 모르니
    if len(candidate_idx) == 0:
        best_idx = int(np.argmax(widths_s))
        stable_width = float(widths_s[best_idx])
        
        return {
            "max_d": float(widths_s[best_idx]), #정맥/동맥 나누는 기준
            "max_x": int(xs[best_idx]), #그 두께를 잰 지점의 좌표
            "max_y": int(ys[best_idx]), #그 지점에서 자가 놓인 각도
            "local_perp_v": pvs[best_idx],  #왼쪽 오른쪽 각각의 거리
            "d_pos": float(d_pos_list[best_idx]),  # 비대칭 그리기용, 스켈레톤의 양옆이 항상 동일할 수는 없으니
            "d_neg": float(d_neg_list[best_idx]),  # 비대칭 그리기용
            "widths_raw": widths_raw,   # 전체 두께 배열
            "widths_s": widths_s,
            "plateau_idx": [best_idx],
        }

    groups = []
    current = [candidate_idx[0]]
    for i in candidate_idx[1:]:
        if i == current[-1] + 1:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    valid_groups = [g for g in groups if len(g) >= min_plateau_len] #혈관의 최대 직경 구간이라면 최소한 3픽셀 
                                                                    #정도는 꾸준히 굵어야 진짜 혈관 직경으로 
    if len(valid_groups) == 0:
        valid_groups = groups

    best_group = max(valid_groups, key=lambda g: (len(g), np.mean(widths_s[g])))
    center_idx = best_group[len(best_group) // 2]   #대표위치 좌표로 씀.
    stable_width = float(np.mean(widths_s[best_group]))

    return {
        "max_d": stable_width,
        "max_x": int(xs[center_idx]),
        "max_y": int(ys[center_idx]),
        "local_perp_v": pvs[center_idx],
        "d_pos": float(d_pos_list[center_idx]),  # 비대칭 그리기용
        "d_neg": float(d_neg_list[center_idx]),  # 비대칭 그리기용
        "widths_raw": widths_raw,
        "widths_s": widths_s,
        "plateau_idx": best_group,
    }

''''
1. 전처리
2. 좌표 매핑
3. 구조 분리
4. 측정 및 분리


'''
def analyze_single_image(image_path, df_keypoints):
    img = read_image_unicode(image_path)
    if img is None:
        return {"ok": False, "reason": f"이미지를 읽지 못했습니다: {image_path}"}

    #분할된 혈관 내에 있어서 각 지점의 두께를 측정하기 위해 범위 구간 정해둠.
    binary_image = img > 127
    skeleton = skeletonize(binary_image)    #스켈레톤화

    labeled_full = label(skeleton)
    if labeled_full.max() == 0:
        return {"ok": False, "reason": "스켈레톤이 없습니다."}
    
    # 노이즈 제거 -> 제일 큰 뼈대 하나만 남김
    counts = np.bincount(labeled_full.flat) #라벨 번호가 몇 픽셀씩 있는지 세는 코드
    main_label = np.argmax(counts[1:]) + 1  #counts는 배경이라서 제외하고
    skeleton = (labeled_full == main_label)

    #어느 혈관을 어떤 방향으로 어디서부터 볼지를 정하는 단계.
    search_name = os.path.splitext(os.path.basename(image_path))[0] #전체 경로에서 파일명만 뽑음
    row = df_keypoints[df_keypoints["filename"].apply(lambda x: os.path.splitext(str(x))[0]) == search_name]    #확장자 제거함.
    if row.empty:
        return {"ok": False, "reason": f"{search_name} 에 해당하는 CSV 행이 없습니다."}

    ux, uy = float(row["U_x"].values[0]), float(row["U_y"].values[0])
    dx, dy = float(row["D_x"].values[0]), float(row["D_y"].values[0])

    U_xy = (ux, uy)
    D_xy = (dx, dy)

    # 스켈레톤위의 출발점과 전체적인 방향을 찾는 과정 단계
    vec_axis = np.array([dx - ux, dy - uy], dtype=float)
    norm_axis = np.linalg.norm(vec_axis)
    dir_v = np.array([0.0, 1.0]) if norm_axis < 1e-5 else vec_axis / norm_axis

    #분석 구간과 시작점과 끝점을 확정하는 단계 
    # -> MTL에서 학습해서 나온 결과 U,D좌표를 실제 스켈레톤 위로 스냅핑하는 과정과, U-D 사이의 구간에서 양쪽 다리의 시작점을 찾는 과정.
    apex_cut_pt = snap_xy_to_skeleton(skeleton, U_xy)
    
    used_branch_pt = find_bottom_of_inner_hole(binary_image, D_xy, dir_v)

    # 혈관 구조 쪼개버림
    split_result = extract_two_leg_paths(skeleton, apex_cut_pt, U_xy, D_xy, used_branch_pt)
    if split_result is None:
        return {"ok": False, "reason": "두 다리 path 추출에 실패했습니다."}
    used_branch_pt = split_result["branch_pt"]
    cut_skeleton = split_result["cut_skeleton"]
    raw_paths = split_result["paths"]
    left_seed = split_result.get("left_seed", None)
    right_seed = split_result.get("right_seed", None)
    left_mask = split_result.get("left_mask", None)
    right_mask = split_result.get("right_mask", None)
    left_vessel_mask, right_vessel_mask = build_vessel_masks_from_side_skeletons(
    binary_image,
    left_mask,
    right_mask
)

    # 시각화, 왼쪽 다리에는 1, 오른쪽 다리에는 2
    labeled_mask = np.zeros_like(binary_image, dtype=np.uint8)
    if left_vessel_mask is not None:
        labeled_mask[left_vessel_mask] = 1
    if right_vessel_mask is not None:
        labeled_mask[right_vessel_mask] = 2

    leg_results = []

    '''
    path는 원래 전체 다리
    trimmed_path는 실제 측정용 다듬은 다리 
    '''
    # 각 다리의 지점마다 두께 측정
    for idx, path in enumerate(raw_paths):
        if len(path) < 2:
            continue

        trimmed_path, start_pt, end_pt, _ = trim_path_for_measurement(
            path, U_xy, D_xy, used_branch_pt, min_keep=5, pixel_margin=0.5,
        )
        if len(trimmed_path) == 0:
            continue

        # 두께 측정 부분 
        measure_mask = left_vessel_mask if idx == 0 else right_vessel_mask
        if measure_mask is None:
            continue

        stable_res = get_stable_max_diameter_raycast(
            trimmed_path,
            measure_mask,
            dir_v,
            D_xy,
            smooth_k=3,
            top_ratio=0.15,
            min_plateau_len=3,
        )
        if stable_res is None:
            continue

        trimmed_xs = np.array([p[1] for p in trimmed_path], dtype=float)
        
        leg_results.append({
            "id": idx,
            "max_d": float(stable_res["max_d"]),    #최대 직경
            "max_x": int(stable_res["max_x"]),
            "max_y": int(stable_res["max_y"]),
            "max_d_pos": stable_res["d_pos"],    
            "max_d_neg": stable_res["d_neg"],    
            "local_perp_v": stable_res["local_perp_v"], 
            "mean_x": float(np.mean(trimmed_xs)),
            "path": path,
            "trimmed_path": trimmed_path,
            "start_pt": start_pt,
            "end_pt": end_pt,
            "radii_raw": stable_res["widths_raw"],
            "radii_s": stable_res["widths_s"],
            "plateau_idx": stable_res["plateau_idx"],
        })

    if len(leg_results) == 0:
        return {"ok": False, "reason": "유효한 limb 결과가 없습니다."}

    leg_results.sort(key=lambda x: x["mean_x"])
    if len(leg_results) >= 2:
        #굵기 순으로 정렬해서 세정맥과 세동맥을 구분
        by_size = sorted(range(len(leg_results)), key=lambda i: leg_results[i]["max_d"], reverse=True)
        leg_results[by_size[0]]["final_label"] = "Venous"
        leg_results[by_size[1]]["final_label"] = "Arterial"

    return {
        "ok": True,
        "image_path": image_path,
        "search_name": search_name,
        "img": img,
        "binary_image": binary_image,
        "skeleton": skeleton,
        "cut_skeleton": cut_skeleton,
        "labeled_mask": labeled_mask,
        "leg_results": leg_results,
        "U_xy": U_xy,
        "D_xy": D_xy,
        "used_branch_pt": used_branch_pt,
        "dir_v": dir_v,
        "left_seed": left_seed,
        "right_seed": right_seed,
        "apex_cut_pt": apex_cut_pt,
    }