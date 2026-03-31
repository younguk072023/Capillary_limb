import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import skeletonize

# [수정] smooth_1d 임포트 추가
from utils import read_image_unicode, snap_xy_to_skeleton, smooth_1d 
from core_algorithm import (
    find_bottom_of_inner_hole,
    extract_two_leg_paths,
    trim_path_for_measurement
)

# --- [새로 추가된 핵심 엔진: Ray Casting 기반 직경 측정] ---
def get_stable_max_diameter_raycast(trimmed_path, binary_image, dir_v, smooth_k=5, top_ratio=0.15, min_plateau_len=3):
    if len(trimmed_path) == 0:
        return None

    ys = np.array([p[0] for p in trimmed_path])
    xs = np.array([p[1] for p in trimmed_path])
    h, w = binary_image.shape

    widths_raw = np.zeros(len(trimmed_path))
    pvs = []
    d_pos_list = []
    d_neg_list = []

    for i in range(len(trimmed_path)):
        # 로컬 방향(Tangent) 계산
        t_start = max(0, i - 3)
        t_end = min(len(trimmed_path) - 1, i + 3)
        
        dx_loc = xs[t_end] - xs[t_start]
        dy_loc = ys[t_end] - ys[t_start]
        norm_loc = np.hypot(dx_loc, dy_loc)
        
        if norm_loc < 1e-6:
            pv = np.array([-dir_v[1], dir_v[0]], dtype=float)
        else:
            pv = np.array([-dy_loc, dx_loc], dtype=float) / norm_loc
        
        # 1. 양의 방향(Positive)으로 Ray 쏘기
        d_pos = 0.0
        for step in range(1, 150):
            ny = int(round(ys[i] + pv[1] * step))
            nx = int(round(xs[i] + pv[0] * step))
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not binary_image[ny, nx]:
                d_pos = np.hypot(nx - xs[i], ny - ys[i])
                break
        
        # 2. 음의 방향(Negative)으로 Ray 쏘기
        d_neg = 0.0
        for step in range(1, 150):
            ny = int(round(ys[i] - pv[1] * step))
            nx = int(round(xs[i] - pv[0] * step))
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not binary_image[ny, nx]:
                d_neg = np.hypot(nx - xs[i], ny - ys[i])
                break
        
        # 양방향 거리의 합이 실제 단면적(Diameter)
        widths_raw[i] = d_pos + d_neg
        pvs.append(pv)
        d_pos_list.append(d_pos)
        d_neg_list.append(d_neg)

    # 지름 데이터 스무딩 및 안정적인 최대값(Plateau) 찾기
    widths_s = smooth_1d(widths_raw, k=smooth_k)
    thr = np.quantile(widths_s, 1.0 - top_ratio)
    candidate_idx = np.where(widths_s >= thr)[0]

    if len(candidate_idx) == 0:
        best_idx = int(np.argmax(widths_s))
        return {
            "max_d": float(widths_s[best_idx]),
            "max_x": int(xs[best_idx]),
            "max_y": int(ys[best_idx]),
            "local_perp_v": pvs[best_idx],
            "d_pos": float(d_pos_list[best_idx]),  # 비대칭 그리기용
            "d_neg": float(d_neg_list[best_idx]),  # 비대칭 그리기용
            "widths_raw": widths_raw,
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

    valid_groups = [g for g in groups if len(g) >= min_plateau_len]
    if len(valid_groups) == 0:
        valid_groups = groups

    best_group = max(valid_groups, key=lambda g: (len(g), np.mean(widths_s[g])))
    center_idx = best_group[len(best_group) // 2]
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
# --------------------------------------------------------

def analyze_single_image(image_path, df_keypoints):
    img = read_image_unicode(image_path)
    if img is None:
        return {"ok": False, "reason": f"이미지를 읽지 못했습니다: {image_path}"}

    binary_image = img > 127
    skeleton = skeletonize(binary_image)

    labeled_full = label(skeleton)
    if labeled_full.max() == 0:
        return {"ok": False, "reason": "스켈레톤이 없습니다."}

    counts = np.bincount(labeled_full.flat)
    main_label = np.argmax(counts[1:]) + 1
    skeleton = (labeled_full == main_label)

    search_name = os.path.splitext(os.path.basename(image_path))[0]
    row = df_keypoints[df_keypoints["filename"].apply(lambda x: os.path.splitext(str(x))[0]) == search_name]
    if row.empty:
        return {"ok": False, "reason": f"{search_name} 에 해당하는 CSV 행이 없습니다."}

    ux, uy = float(row["U_x"].values[0]), float(row["U_y"].values[0])
    dx, dy = float(row["D_x"].values[0]), float(row["D_y"].values[0])

    U_xy = (ux, uy)
    D_xy = (dx, dy)

    vec_axis = np.array([dx - ux, dy - uy], dtype=float)
    norm_axis = np.linalg.norm(vec_axis)
    dir_v = np.array([0.0, 1.0]) if norm_axis < 1e-5 else vec_axis / norm_axis

    apex_cut_pt = snap_xy_to_skeleton(skeleton, U_xy)
    used_branch_pt = find_bottom_of_inner_hole(binary_image, D_xy, dir_v)

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

    labeled_mask = np.zeros_like(binary_image, dtype=np.uint8)
    if left_mask is not None:
        labeled_mask[left_mask] = 1
    if right_mask is not None:
        labeled_mask[right_mask] = 2

    leg_results = []
    for idx, path in enumerate(raw_paths):
        if len(path) < 2:
            continue

        trimmed_path, start_pt, end_pt, _ = trim_path_for_measurement(
            path, U_xy, D_xy, used_branch_pt, min_keep=5, pixel_margin=0.5,
        )
        if len(trimmed_path) == 0:
            continue

        # [수정] Ray Casting 알고리즘으로 교체!
        stable_res = get_stable_max_diameter_raycast(
            trimmed_path,
            binary_image,
            dir_v,
            smooth_k=5,
            top_ratio=0.15,
            min_plateau_len=3,
        )
        if stable_res is None:
            continue

        trimmed_xs = np.array([p[1] for p in trimmed_path], dtype=float)
        
        leg_results.append({
            "id": idx,
            "max_d": float(stable_res["max_d"]),
            "max_x": int(stable_res["max_x"]),
            "max_y": int(stable_res["max_y"]),
            # [추가] 양쪽 비대칭 거리를 각각 저장
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