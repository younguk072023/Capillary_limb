import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import skeletonize

from utils import read_image_unicode, snap_xy_to_skeleton
from core_algorithm import (
    find_bottom_of_inner_hole,
    extract_two_leg_paths,
    trim_path_for_measurement,
    get_stable_max_diameter
)

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

    dist_map = distance_transform_edt(binary_image)

    leg_results = []
    for idx, path in enumerate(raw_paths):
        if len(path) < 2:
            continue

        trimmed_path, start_pt, end_pt, _ = trim_path_for_measurement(
            path,
            U_xy,
            D_xy,
            used_branch_pt,
            min_keep=5,
            pixel_margin=0.5,
        )
        if len(trimmed_path) == 0:
            continue

        stable_res = get_stable_max_diameter(
            trimmed_path,
            dist_map,
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
            "mean_x": float(np.mean(trimmed_xs)),
            "path": path,
            "trimmed_path": trimmed_path,
            "start_pt": start_pt,
            "end_pt": end_pt,
            "radii_raw": stable_res["radii_raw"],
            "radii_s": stable_res["radii_s"],
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