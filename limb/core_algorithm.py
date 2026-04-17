import cv2
import numpy as np
from utils import (
    get_neighbors,
    geodesic_distances_from_seed,
    reconstruct_path,
    keep_component_containing_seed,
    find_endpoints,
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
def sample_perp_line_points(center_xy, perp_v, half_len=50, num=121):
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
def find_left_right_seed_on_crossline(skeleton, center_xy, perp_v, img_shape, min_sep=1, band_height=20):
    h, w = img_shape

    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return None, None, None

    perp_v = np.asarray(perp_v, dtype=float)
    perp_v = perp_v / (np.linalg.norm(perp_v) + 1e-12)

    # perp_v에 수직인 방향 = D line 아래쪽 방향
    dir_v = np.array([perp_v[1], -perp_v[0]], dtype=float)

    C = np.array([center_xy[0], center_xy[1]], dtype=float)
    P = np.stack([xs, ys], axis=1).astype(float)   # (x, y)

    rel = P - C[None, :]
    t = rel @ perp_v   # line 방향 좌표 (좌/우 구분용)
    n = rel @ dir_v    # line 아래쪽 거리

    # 빨간선 바로 아래쪽 band 안만 사용
    valid = (n >= 0) & (n <= band_height)

    left_idx = np.where(valid & (t < 0))[0]
    right_idx = np.where(valid & (t > 0))[0]

    if len(left_idx) == 0 or len(right_idx) == 0:
        return None, None, None

    # 각 쪽에서 빨간선에 가장 가까운 점 우선, 같으면 더 바깥쪽 점 우선
    left_order = np.lexsort((-np.abs(t[left_idx]), n[left_idx]))
    right_order = np.lexsort((-np.abs(t[right_idx]), n[right_idx]))

    li = left_idx[left_order[0]]
    ri = right_idx[right_order[0]]

    left_seed = (int(P[li, 1]), int(P[li, 0]))    # (y, x)
    right_seed = (int(P[ri, 1]), int(P[ri, 0]))   # (y, x)

    score = float(abs(t[li] - t[ri]))
    if score < min_sep:
        return None, None, None

    return left_seed, right_seed, score

def debug_crossline_at_center(skeleton, center_xy, perp_v, img_shape, min_sep=1, band_height=20):
    h, w = img_shape

    # 빨간선 자체 샘플(화면용 hit)
    half_len = int(np.ceil(np.hypot(h, w)))
    num = max(141, 2 * half_len + 1)
    pts = sample_perp_line_points(center_xy, perp_v, half_len=half_len, num=num)

    hits = []
    for t, y, x in pts:
        if 0 <= y < h and 0 <= x < w and skeleton[y, x]:
            hits.append((t, y, x))

    # 실제 seed는 near-line 방식으로 계산
    left_seed, right_seed, score = find_left_right_seed_on_crossline(
        skeleton=skeleton,
        center_xy=center_xy,
        perp_v=perp_v,
        img_shape=img_shape,
        min_sep=min_sep,
        band_height=band_height
    )

    return {
        "center_xy": center_xy,
        "pts": pts,
        "hits": hits,
        "clusters": [],
        "left_seed": left_seed,
        "right_seed": right_seed,
        "score": score,
    }

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


from scipy.ndimage import distance_transform_edt

def build_vessel_masks_from_side_skeletons(binary_image, left_mask, right_mask):
    if left_mask is None or right_mask is None:
        return None, None

    vessel = binary_image.astype(bool)

    # 각 픽셀이 왼쪽 skeleton / 오른쪽 skeleton 중 어디에 더 가까운지 계산
    left_dist = distance_transform_edt(~left_mask)
    right_dist = distance_transform_edt(~right_mask)

    left_vessel_mask = vessel & (left_dist <= right_dist)
    right_vessel_mask = vessel & (right_dist < left_dist)

    return left_vessel_mask, right_vessel_mask
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

