import os
import numpy as np

from skimage.measure import label
from skimage.morphology import skeletonize
from utils import read_image_unicode, snap_xy_to_skeleton
from core_algorithm import (
    find_bottom_of_inner_hole,
    extract_two_leg_paths,
    trim_path_for_measurement,
    build_vessel_masks_from_side_skeletons,
    debug_crossline_at_center,
    build_cut_skeleton,
)

'''
한 개의 limb path가 주어졌을 때,
그 path를 따라가며 각 지점의 두께를 재고
그중에서 가장 안정적인 최대 직경을 뽑는 함수
'''

"""
혈관 직경 측정 - 강건(robust) 버전
이상치 제거를 위해 다음 5가지 필터를 통합:
  1) EDT(Euclidean Distance Transform) 교차검증
  2) 광선 대칭성 검사 (d_pos ≈ d_neg)
  3) 곡률 기반 고곡률 구간 제외
  4) Savitzky-Golay 스무딩
  5) 생리학적 상한 (선택적)

각 필터는 개별 파라미터로 on/off 및 임계값 조정 가능.
논문 디펜스/리뷰 대응을 위해 'reject_reasons'로 각 점의 탈락 사유를 기록.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter


"""
혈관 직경 측정 (디펜스 가능한 단순 버전)

핵심 아이디어 하나:
    Raycast로 잰 폭(widths_raw)과
    EDT로 잰 내접원 지름(2 * EDT)은
    접선 추정이 정확한 지점에서만 서로 일치한다.
    → 두 값의 비율이 1에서 너무 벗어난 지점은 측정 오류로 간주하고 제외.

이 한 가지 원리만으로 꺾인 혈관에서 발생하던 이상치(예: 142px)를
정상값으로 되돌린다.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def get_stable_max_diameter_raycast(
    trimmed_path,
    binary_image,
    dir_v,
    D_xy,
    branch_pt=None,
    tangent_half_window=3,
    use_branch_ray_gate=True,
    # --- 유일한 튜닝 파라미터 ---
    edt_ratio_max=1.5,          # raycast가 EDT 지름의 몇 배까지 허용할지
    max_diameter_px=None,       # (선택) 생리학적 상한. 모르면 None
):
    """
    각 지점에서 raycast 폭을 재고, EDT 참조값과 비교해서
    접선 추정이 제대로 된 지점 중에서만 최대값을 고른다.

    Parameters
    ----------
    edt_ratio_max : float, 기본 1.5
        raycast_width / (2 * EDT)가 이 값을 넘으면 해당 지점 제외.
        이론상 수직 측정이면 비율이 1.0 근처여야 하며,
        이산화(정수 픽셀 반올림)와 약간의 접선 오차로 최대 ~1.3까지는 정상.
        1.5는 "정상보다 50% 이상 부풀려진 측정"을 잘라낸다는 뜻.

    Returns
    -------
    dict with keys:
        max_d, max_x, max_y, local_perp_v, d_pos, d_neg,
        widths_raw, widths_s, edt_diameters,
        valid_mask, n_total, n_valid, plateau_idx
    """
    N = len(trimmed_path)
    if N == 0:
        return None

    ys = np.array([p[0] for p in trimmed_path], dtype=float)
    xs = np.array([p[1] for p in trimmed_path], dtype=float)
    h, w = binary_image.shape

    B_vec = None
    if branch_pt is not None:
        B_vec = np.array([branch_pt[1], branch_pt[0]], dtype=float)

    D_dir = np.array([dir_v[0], dir_v[1]], dtype=float)

    # ===========================================================
    # [1] 각 지점에서 접선 방향 추정 → 그에 수직인 방향 pv 계산
    # ===========================================================
    # pv가 혈관 축에 수직이어야 raycast가 '두께'를 측정.
    # 혈관이 꺾이면 이 추정이 틀려서 pv가 혈관축을 따라 눕는다 → 이상치 발생.
    pvs = np.zeros((N, 2))
    for i in range(N):
        t_s = max(0, i - tangent_half_window)
        t_e = min(N - 1, i + tangent_half_window)
        dx_loc = xs[t_e] - xs[t_s]
        dy_loc = ys[t_e] - ys[t_s]
        norm_loc = np.hypot(dx_loc, dy_loc)
        if norm_loc < 1e-6:
            tx, ty = D_dir[0], D_dir[1]
        else:
            tx, ty = dx_loc / norm_loc, dy_loc / norm_loc
        pvs[i] = [-ty, tx]   # 접선을 90도 회전 = 수직 벡터

    # ===========================================================
    # [2] 각 지점에서 pv 방향으로 양쪽으로 광선을 쏴 경계까지 거리 측정
    # ===========================================================
    widths_raw = np.zeros(N)
    d_pos_arr = np.zeros(N)
    d_neg_arr = np.zeros(N)

    for i in range(N):
        pv = pvs[i]
        d_pos = _cast_ray(xs[i], ys[i], pv, +1, binary_image, h, w,
                          B_vec, D_dir, use_branch_ray_gate)
        d_neg = _cast_ray(xs[i], ys[i], pv, -1, binary_image, h, w,
                          B_vec, D_dir, use_branch_ray_gate)
        widths_raw[i] = d_pos + d_neg
        d_pos_arr[i] = d_pos
        d_neg_arr[i] = d_neg

    # ===========================================================
    # [3] EDT로 "각 지점의 내접원 지름" 계산 (접선과 무관한 독립 지표)
    # ===========================================================
    # EDT[y, x] = (y, x)에서 혈관 경계까지의 최단거리.
    # 2 * EDT = 그 점을 중심으로 혈관 안에 들어갈 수 있는 최대 원의 지름.
    # 이 값은 수직이 맞든 틀리든 항상 "국소 두께"의 정답에 가깝다.
    edt = distance_transform_edt(binary_image)
    yi = np.clip(ys.astype(int), 0, h - 1)
    xi = np.clip(xs.astype(int), 0, w - 1)
    edt_diameters = 2.0 * edt[yi, xi]

    # ===========================================================
    # [4] 유효점 판정: raycast가 EDT 지름보다 너무 크면 접선 오정렬
    # ===========================================================
    # 정상 측정: widths_raw ≈ 2*EDT (비율 1.0 근처)
    # 꺾인 혈관에서 pv 오정렬: widths_raw >> 2*EDT (비율이 2~5배로 튐)
    ratio = widths_raw / np.maximum(edt_diameters, 1e-6)
    valid = ratio <= edt_ratio_max

    # (선택) 문헌 기반 절대 상한 — 알고 있으면 추가 방어선
    if max_diameter_px is not None:
        valid &= (widths_raw <= max_diameter_px)

    # 전부 탈락하면 EDT 자체를 폴백으로 사용 (raycast 포기)
    # 이 경우 "이 다리는 전 구간이 꺾임" → EDT 최대값이 가장 합리적 추정
    if not valid.any():
        best_idx = int(np.argmax(edt_diameters))
        return _build_result(
            best_idx, xs, ys, pvs, widths_raw, d_pos_arr, d_neg_arr,
            edt_diameters, valid, used_fallback=True,
        )

    # ===========================================================
    # [5] 유효점 중에서 최대값 선택
    # ===========================================================
    widths_masked = np.where(valid, widths_raw, -np.inf)
    best_idx = int(np.argmax(widths_masked))

    return _build_result(
        best_idx, xs, ys, pvs, widths_raw, d_pos_arr, d_neg_arr,
        edt_diameters, valid, used_fallback=False,
    )


# -----------------------------------------------------------------
# 내부 유틸 함수 두 개 (함수 본체를 짧게 유지하려고 분리)
# -----------------------------------------------------------------
def _cast_ray(x0, y0, pv, sign, binary_image, h, w,
              B_vec, D_dir, use_branch_gate, max_step=150):
    """한 방향(+ 또는 -)으로 광선을 쏴서 경계를 만날 때까지의 거리."""
    for step in range(1, max_step):
        nx = int(round(x0 + sign * pv[0] * step))
        ny = int(round(y0 + sign * pv[1] * step))

        # 분기점 게이트: branch_pt 아래쪽으로는 광선이 넘어가지 못하게
        if use_branch_gate and B_vec is not None:
            P_ray = np.array([nx, ny], dtype=float)
            if np.dot(P_ray - B_vec, D_dir) > 0:
                return np.hypot(nx - x0, ny - y0)

        # 이미지 밖이거나 경계를 벗어나면 거리 반환
        if ny < 0 or ny >= h or nx < 0 or nx >= w:
            return np.hypot(nx - x0, ny - y0)
        if not binary_image[ny, nx]:
            return np.hypot(nx - x0, ny - y0)

    return np.hypot(sign * pv[0] * max_step, sign * pv[1] * max_step)


def _build_result(best_idx, xs, ys, pvs, widths_raw, d_pos_arr, d_neg_arr,
                  edt_diameters, valid, used_fallback):
    """반환 dict 구성. 폴백 모드면 EDT 기반 폭을 max_d로 사용."""
    if used_fallback:
        final_d = float(edt_diameters[best_idx])
        # 폴백 시 d_pos/d_neg는 EDT의 절반씩 분배(시각화용 근사)
        d_pos = d_neg = final_d / 2.0
    else:
        final_d = float(widths_raw[best_idx])
        d_pos = float(d_pos_arr[best_idx])
        d_neg = float(d_neg_arr[best_idx])

    return {
        "max_d": final_d,
        "max_x": int(xs[best_idx]),
        "max_y": int(ys[best_idx]),
        "local_perp_v": pvs[best_idx],
        "d_pos": d_pos,
        "d_neg": d_neg,
        "widths_raw": widths_raw,
        "widths_s": widths_raw.copy(),        # viewer 호환용
        "edt_diameters": edt_diameters,
        "valid_mask": valid,
        "n_total": len(widths_raw),
        "n_valid": int(valid.sum()),
        "plateau_idx": [best_idx],
        "used_edt_fallback": used_fallback,
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

    perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

    
    preview_cut_skeleton, _ = build_cut_skeleton(
        skeleton,
        apex_cut_pt,
        used_branch_pt,
        dilate_size=1
    )

    debug_dline = debug_crossline_at_center(
        skeleton=preview_cut_skeleton,
        center_xy=D_xy,
        perp_v=perp_v,
        img_shape=preview_cut_skeleton.shape,
        #min_sep=1,
        band_height=20
    )

    # 혈관 구조 쪼개버림
    split_result = extract_two_leg_paths(skeleton, apex_cut_pt, U_xy, D_xy, used_branch_pt)
    if split_result is None:
        return {
            "ok": False,
            "reason": "두 다리 path 추출에 실패했습니다.",
            "img": img,
            "binary_image": binary_image,
            "skeleton": skeleton,
            "U_xy": U_xy,
            "D_xy": D_xy,
            "dir_v": dir_v,
            "apex_cut_pt": apex_cut_pt,
            "used_branch_pt": used_branch_pt,
            "debug_dline": debug_dline,
            "search_name": search_name,
        }
    
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
            branch_pt=used_branch_pt,
            tangent_half_window=3,
            use_branch_ray_gate=True,
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