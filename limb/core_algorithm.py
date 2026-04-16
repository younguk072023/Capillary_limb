import cv2
import numpy as np
from utils import (
    get_neighbors,
    geodesic_distances_from_seed,
    reconstruct_path,
    keep_component_containing_seed,
    find_endpoints,
    smooth_1d
)
# 자를 위치 표시
def build_cut_skeleton(skeleton, apex_cut_pt, branch_pt=None, dilate_size=3):
    cut_mask = np.zeros_like(skeleton, dtype=np.uint8)

    if apex_cut_pt is not None:
        ay, ax = apex_cut_pt
        cut_mask[ay, ax] = 1
        if dilate_size > 0:
            cv2.circle(cut_mask, (ax, ay), dilate_size, 1, -1)

    if branch_pt is not None:
        by, bx = branch_pt
        cut_mask[by, bx] = 1
        if dilate_size > 0:
            cv2.circle(cut_mask, (bx, by), dilate_size, 1, -1)

    cut_skeleton = skeleton.copy()
    cut_skeleton[cut_mask > 0] = False
    return cut_skeleton, cut_mask

# 혈관의 분기점이있을 경우 구명의 바닥 부분 찾는 함수
def find_bottom_of_inner_hole(binary_image, D_xy, dir_v):
    h, w = binary_image.shape
    bg_mask = (~binary_image).astype(np.uint8)

    seed_x, seed_y = None, None
    for k in range(3, 30):
        sx = int(round(D_xy[0] + dir_v[0] * k))
        sy = int(round(D_xy[1] + dir_v[1] * k))
        if 0 <= sx < w and 0 <= sy < h and bg_mask[sy, sx] == 1:
            seed_x, seed_y = sx, sy
            break

    if seed_x is None:
        return None

    _, labels = cv2.connectedComponents(bg_mask)
    hole_label = labels[seed_y, seed_x]

    hole_ys, hole_xs = np.where(labels == hole_label)
    if len(hole_ys) > (h * w * 0.4):
        return None

    true_bottom_pt = None
    max_proj_dist = -float("inf")

    for hy, hx in zip(hole_ys, hole_xs):
        proj = np.dot(np.array([hx - D_xy[0], hy - D_xy[1]], dtype=float), dir_v)
        if proj > max_proj_dist:
            max_proj_dist = proj
            true_bottom_pt = (int(hy), int(hx))

    return true_bottom_pt

# skeleton의 혈관을 따라가서 양쪽 다리의 가장 큰 길이 구하는 함수
def sample_perp_line_points(center_xy, perp_v, half_len=24, num=121):
    cx, cy = center_xy
    ts = np.linspace(-half_len, half_len, num)
    out = []
    last = None
    for t in ts:
        x = int(round(cx + perp_v[0] * t))
        y = int(round(cy + perp_v[1] * t))
        if last != (y, x):
            out.append((float(t), int(y), int(x)))
            last = (y, x)
    return out

#왼쪽 다리와 오른쪽 다리를 처음으로 구분해주는 seed찾는 함수
def find_left_right_seed_on_crossline(skeleton, center_xy, perp_v, img_shape, min_sep=3):
    h, w = img_shape
    pts = sample_perp_line_points(center_xy, perp_v, half_len=28, num=141)

    hits = []
    for t, y, x in pts:
        if 0 <= y < h and 0 <= x < w and skeleton[y, x]:
            hits.append((t, y, x))

    if len(hits) < 2:
        return None, None, None

    clusters = []
    current = [hits[0]]
    for item in hits[1:]:
        prev = current[-1]
        if abs(item[0] - prev[0]) <= 1.5:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    if len(clusters) < 2:
        return None, None, None

    left_cluster = clusters[-1]
    right_cluster = clusters[0]

    left_seed = left_cluster[len(left_cluster) // 2][1:]
    right_seed = right_cluster[len(right_cluster) // 2][1:]

    left_t = left_cluster[len(left_cluster) // 2][0]
    right_t = right_cluster[len(right_cluster) // 2][0]

    if abs(right_t - left_t) < min_sep:
        return None, None, None

    score = abs(right_t - left_t)
    return left_seed, right_seed, score

# apex width을 찾아주는 함수
def find_two_leg_seeds_between_U_and_D(skeleton, U_xy, D_xy):
    Ux, Uy = U_xy
    Dx, Dy = D_xy

    axis_v = np.array([Dx - Ux, Dy - Uy], dtype=float)
    norm = np.linalg.norm(axis_v)
    if norm < 1e-6:
        return None

    # U->D 방향
    dir_v = axis_v / norm

    # apex width line 방향 = U->D에 수직
    perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

    # 네 방식: D점 한 군데에서만 좌/우 seed 찾기
    left_seed, right_seed, score = find_left_right_seed_on_crossline(
        skeleton=skeleton,
        center_xy=(Dx, Dy),          # D점 기준
        perp_v=perp_v,               # apex width line 방향
        img_shape=skeleton.shape,
        min_sep=3
    )

    if left_seed is None or right_seed is None:
        return None

    return left_seed, right_seed

# 두 점을 이용해서 전체 혈관 벼대를 왼쪽과 오른쪽으로 구분해주는 함수
def build_side_masks_from_two_seeds(cut_skeleton, left_seed, right_seed):
    left_dist, _ = geodesic_distances_from_seed(cut_skeleton, left_seed)
    right_dist, _ = geodesic_distances_from_seed(cut_skeleton, right_seed)

    left_mask = np.zeros_like(cut_skeleton, dtype=bool)
    right_mask = np.zeros_like(cut_skeleton, dtype=bool)

    ys, xs = np.where(cut_skeleton)
    for y, x in zip(ys, xs):
        p = (int(y), int(x))
        dl = left_dist.get(p, np.inf)
        dr = right_dist.get(p, np.inf)

        if np.isinf(dl) and np.isinf(dr):
            continue
        if dl < dr:
            left_mask[y, x] = True
        elif dr < dl:
            right_mask[y, x] = True
        else:
            e_left = (x - left_seed[1]) ** 2 + (y - left_seed[0]) ** 2
            e_right = (x - right_seed[1]) ** 2 + (y - right_seed[0]) ** 2
            if e_left <= e_right:
                left_mask[y, x] = True
            else:
                right_mask[y, x] = True

    left_mask[left_seed[0], left_seed[1]] = True
    right_mask[right_seed[0], right_seed[1]] = True

    left_mask = keep_component_containing_seed(left_mask, left_seed)
    right_mask = keep_component_containing_seed(right_mask, right_seed)

    return left_mask, right_mask


# 혈관의 끝점과 apex 사이의 경로를 자르는 함수 (측정에 필요한 부분만 남기는 함수)
def choose_best_endpoint_for_leg(side_mask, seed_pt, D_xy, dir_v):
    dist, parent = geodesic_distances_from_seed(side_mask, seed_pt)
    if len(dist) == 0:
        return None, [], dist, parent

    endpoints = [p for p in find_endpoints(side_mask) if p != seed_pt]
    if len(endpoints) == 0:
        endpoints = [p for p in dist.keys() if p != seed_pt]
        if len(endpoints) == 0:
            return None, [], dist, parent

    positive_candidates = []
    fallback_candidates = []

    for ep in endpoints:
        proj = float(np.dot(np.array([ep[1] - D_xy[0], ep[0] - D_xy[1]], dtype=float), dir_v))
        path_len = dist.get(ep, 0)
        score = 6.0 * proj + 1.0 * path_len
        fallback_score = 1.0 * path_len + 0.25 * proj
        fallback_candidates.append((fallback_score, ep))
        if proj > 0.5:
            positive_candidates.append((score, ep))

    if len(positive_candidates) > 0:
        positive_candidates.sort(key=lambda x: x[0], reverse=True)
        best_ep = positive_candidates[0][1]
    else:
        fallback_candidates.sort(key=lambda x: x[0], reverse=True)
        best_ep = fallback_candidates[0][1]

    path = reconstruct_path(parent, best_ep)
    return best_ep, path, dist, parent


def trace_leg_from_seed(side_mask, seed_pt, D_xy, dir_v):
    best_ep, path, _, _ = choose_best_endpoint_for_leg(side_mask, seed_pt, D_xy, dir_v)
    if best_ep is None:
        return []
    return path


def extract_two_leg_paths(skeleton, apex_cut_pt, U_xy, D_xy, branch_pt=None):
    retry_configs = [
        (branch_pt, 3),
        (branch_pt, 2),
        (branch_pt, 1),
        (None, 3),
        (None, 2),
        (None, 1),
    ]

    axis_v = np.array([D_xy[0] - U_xy[0], D_xy[1] - U_xy[1]], dtype=float)
    norm = np.linalg.norm(axis_v)
    dir_v = np.array([0.0, 1.0]) if norm < 1e-6 else axis_v / norm

    best_result = None

    for current_branch, dilate_size in retry_configs:
        cut_skeleton, _ = build_cut_skeleton(skeleton, apex_cut_pt, current_branch, dilate_size)

        seeds = find_two_leg_seeds_between_U_and_D(cut_skeleton, U_xy, D_xy)
        if seeds is None:
            continue

        left_seed, right_seed = seeds
        left_mask, right_mask = build_side_masks_from_two_seeds(cut_skeleton, left_seed, right_seed)

        left_path = trace_leg_from_seed(left_mask, left_seed, D_xy, dir_v)
        right_path = trace_leg_from_seed(right_mask, right_seed, D_xy, dir_v)

        valid_count = int(len(left_path) >= 4) + int(len(right_path) >= 4)

        result = {
            "paths": [left_path, right_path],
            "branch_pt": current_branch,
            "cut_skeleton": cut_skeleton,
            "left_seed": left_seed,
            "right_seed": right_seed,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "valid_count": valid_count,
        }

        if best_result is None or result["valid_count"] > best_result["valid_count"]:
            best_result = result

        if valid_count == 2:
            return result

    return best_result


def trim_path_for_measurement(path, U_xy, D_xy, branch_pt=None, min_keep=5, pixel_margin=0.5):
    vec_axis = np.array([D_xy[0] - U_xy[0], D_xy[1] - U_xy[1]], dtype=float)
    norm = np.linalg.norm(vec_axis)
    dir_v = np.array([0.0, 1.0]) if norm < 1e-5 else vec_axis / norm

    D_vec = np.array([D_xy[0], D_xy[1]], dtype=float)
    B_vec = None
    if branch_pt is not None:
        B_vec = np.array([branch_pt[1], branch_pt[0]], dtype=float)

    trimmed = []
    recording = False

    for pt in path:
        P = np.array([pt[1], pt[0]], dtype=float)
        proj_D = np.dot(P - D_vec, dir_v)

        if not recording and proj_D >= pixel_margin:
            recording = True

        if recording and B_vec is not None:
            proj_B = np.dot(P - B_vec, dir_v)
            if proj_B > -pixel_margin:
                break

        if recording:
            trimmed.append(pt)

    if len(trimmed) < min_keep:
        return [], None, None, dir_v

    return trimmed, trimmed[0], trimmed[-1], dir_v

# 혈관의 두께 측정에서 안정적인 최대값을 찾아주는 함수
def get_stable_max_diameter(trimmed_path, dist_map, smooth_k=5, top_ratio=0.15, min_plateau_len=3):
    if len(trimmed_path) == 0:
        return None

    ys = np.array([p[0] for p in trimmed_path])
    xs = np.array([p[1] for p in trimmed_path])

    radii_raw = dist_map[ys, xs].astype(float)
    radii_s = smooth_1d(radii_raw, k=smooth_k)

    thr = np.quantile(radii_s, 1.0 - top_ratio)
    candidate_idx = np.where(radii_s >= thr)[0]

    if len(candidate_idx) == 0:
        best_idx = int(np.argmax(radii_s))
        return {
            "max_d": float(radii_s[best_idx] * 2.0),
            "max_x": int(xs[best_idx]),
            "max_y": int(ys[best_idx]),
            "radii_raw": radii_raw,
            "radii_s": radii_s,
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

    best_group = max(valid_groups, key=lambda g: (len(g), np.mean(radii_s[g])))
    center_idx = best_group[len(best_group) // 2]
    stable_radius = float(np.mean(radii_s[best_group]))

    return {
        "max_d": stable_radius * 2.0,
        "max_x": int(xs[center_idx]),
        "max_y": int(ys[center_idx]),
        "radii_raw": radii_raw,
        "radii_s": radii_s,
        "plateau_idx": best_group,
    }