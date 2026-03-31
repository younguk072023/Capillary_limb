import os
from collections import deque

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import skeletonize


"""
===========================================================
1. 기본 유틸
===========================================================
"""
def get_neighbors(y, x, mask):
    neighbors = []
    h, w = mask.shape
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors


def smooth_1d(arr, k=5):
    if len(arr) < k or k <= 1:
        return arr.copy()
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(padded, kernel, mode="valid")


def geodesic_distances_from_seed(mask, seed):
    q = deque([seed])
    dist = {seed: 0}
    parent = {seed: None}

    while q:
        cur = q.popleft()
        for nxt in get_neighbors(cur[0], cur[1], mask):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                q.append(nxt)
    return dist, parent


def reconstruct_path(parent, end_pt):
    path = []
    cur = end_pt
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def read_image_unicode(image_path):
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return img


def snap_xy_to_skeleton(skeleton, pt_xy):
    xs = np.where(skeleton)[1]
    ys = np.where(skeleton)[0]
    if len(xs) == 0:
        return None

    px, py = pt_xy
    d2 = (xs - px) ** 2 + (ys - py) ** 2
    idx = int(np.argmin(d2))
    return int(ys[idx]), int(xs[idx])


def keep_component_containing_seed(mask, seed):
    if seed is None or not mask[seed[0], seed[1]]:
        return np.zeros_like(mask, dtype=bool)

    q = deque([seed])
    visited = {seed}
    out = np.zeros_like(mask, dtype=bool)
    out[seed[0], seed[1]] = True

    while q:
        cy, cx = q.popleft()
        for ny, nx in get_neighbors(cy, cx, mask):
            if (ny, nx) not in visited:
                visited.add((ny, nx))
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def find_endpoints(mask):
    ys, xs = np.where(mask)
    endpoints = []
    for y, x in zip(ys, xs):
        deg = len(get_neighbors(y, x, mask))
        if deg <= 1:
            endpoints.append((y, x))
    return endpoints


"""
===========================================================
2. skeleton cut 및 내부 hole 처리
===========================================================
"""
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


"""
===========================================================
3. U-D 사이 구간에서 좌/우 limb 시작점 찾기
===========================================================
"""
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

    left_cluster = clusters[0]
    right_cluster = clusters[-1]

    left_seed = left_cluster[len(left_cluster) // 2][1:]
    right_seed = right_cluster[len(right_cluster) // 2][1:]

    left_t = left_cluster[len(left_cluster) // 2][0]
    right_t = right_cluster[len(right_cluster) // 2][0]

    if abs(right_t - left_t) < min_sep:
        return None, None, None

    score = abs(right_t - left_t)
    return left_seed, right_seed, score


def find_two_leg_seeds_between_U_and_D(skeleton, U_xy, D_xy):
    Ux, Uy = U_xy
    Dx, Dy = D_xy

    axis_v = np.array([Dx - Ux, Dy - Uy], dtype=float)
    norm = np.linalg.norm(axis_v)
    if norm < 1e-6:
        return None

    dir_v = axis_v / norm
    perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

    candidates = []
    for alpha in np.linspace(0.10, 0.90, 17):
        cx = Ux + alpha * (Dx - Ux)
        cy = Uy + alpha * (Dy - Uy)

        left_seed, right_seed, score = find_left_right_seed_on_crossline(
            skeleton=skeleton,
            center_xy=(cx, cy),
            perp_v=perp_v,
            img_shape=skeleton.shape,
            min_sep=3
        )

        if left_seed is not None and right_seed is not None:
            candidates.append((score, alpha, left_seed, right_seed))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: (x[0], -abs(x[1] - 0.5)), reverse=True)
    _, _, left_seed, right_seed = candidates[0]
    return left_seed, right_seed


"""
===========================================================
4. 두 seed 기준 geodesic partition
===========================================================
"""
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


"""
===========================================================
5. 각 side mask 내부에서 leg path 추적
===========================================================
"""
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


"""
===========================================================
6. 순차적 스켈레톤 추적 및 Trim
===========================================================
"""
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


"""
===========================================================
7. stable max diameter
===========================================================
"""
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


"""
===========================================================
8. 단일 이미지 분석 엔진
===========================================================
"""
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


"""
===========================================================
9. 키보드/휠 인터랙티브 뷰어
===========================================================
"""
class CapillaryViewer:
    def __init__(self, image_dir, csv_path):
        self.image_dir = image_dir
        self.csv_path = csv_path

        self.df_keypoints = pd.read_csv(csv_path)

        valid_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_dir}")

        self.index = 0
        self.current_xlim = None
        self.current_ylim = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor="black")
        self.fig.canvas.manager.set_window_title("Capillary Viewer - dual leg robust")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.update_view(reset_zoom=True)
        plt.show()

    def get_current_image_path(self):
        return os.path.join(self.image_dir, self.image_files[self.index])

    def update_view(self, reset_zoom=False):
        self.ax.clear()
        image_path = self.get_current_image_path()
        result = analyze_single_image(image_path, self.df_keypoints)

        if not result["ok"]:
            self.ax.imshow(np.zeros((512, 512)), cmap="gray")
            self.ax.text(
                0.5,
                0.5,
                f"[{self.index + 1}/{len(self.image_files)}]\n{os.path.basename(image_path)}\n\n{result['reason']}",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="red",
                fontsize=12,
            )
            self.ax.axis("off")
            self.fig.tight_layout()
            self.fig.canvas.draw()
            return

        img = result["img"]
        leg_results = result["leg_results"]
        labeled_mask = result["labeled_mask"]
        U_xy = result["U_xy"]
        D_xy = result["D_xy"]
        used_branch_pt = result["used_branch_pt"]
        dir_v = result["dir_v"]
        left_seed = result["left_seed"]
        right_seed = result["right_seed"]
        apex_cut_pt = result["apex_cut_pt"]
        search_name = result["search_name"]

        self.ax.imshow(img, cmap="gray")

        overlay = np.ma.masked_where(labeled_mask == 0, labeled_mask)
        self.ax.imshow(overlay, cmap="nipy_spectral", alpha=0.25)

        result_text = []
        if len(leg_results) >= 2:
            sorted_leg_results = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)
            result_text.append(f"Venous: {sorted_leg_results[0]['max_d']:.2f}px")
            result_text.append(f"Arterial: {sorted_leg_results[1]['max_d']:.2f}px")
        else:
            for i, r in enumerate(leg_results):
                result_text.append(f"Leg{i+1}: {r['max_d']:.2f}px")

        for idx, res in enumerate(leg_results):
            side = "Left" if idx == 0 else "Right"
            line_color = "cyan" if idx == 0 else "yellow"

            px = [p[1] for p in res["path"]]
            py = [p[0] for p in res["path"]]
            self.ax.plot(px, py, color="white", linewidth=1.0, alpha=0.35, zorder=1)

            tx = [p[1] for p in res["trimmed_path"]]
            ty = [p[0] for p in res["trimmed_path"]]
            self.ax.plot(tx, ty, color="black", linewidth=4, zorder=2)
            self.ax.plot(tx, ty, color=line_color, linewidth=2.0, zorder=3, label=f"{side} measured")

            self.ax.scatter(res["max_x"], res["max_y"], c="red", s=70, edgecolors="white", zorder=6)
            self.ax.text(
                res["max_x"],
                res["max_y"] - 12,
                f"{res['max_d']:.2f}",
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
            )

        if left_seed is not None:
            self.ax.scatter(left_seed[1], left_seed[0], c="cyan", s=70, edgecolors="black", marker="o", zorder=7)
        if right_seed is not None:
            self.ax.scatter(right_seed[1], right_seed[0], c="yellow", s=70, edgecolors="black", marker="o", zorder=7)
        if apex_cut_pt is not None:
            self.ax.scatter(apex_cut_pt[1], apex_cut_pt[0], c="magenta", s=80, edgecolors="white", marker="x", zorder=7)

        self.ax.scatter(U_xy[0], U_xy[1], c="gray", s=55, edgecolors="white", zorder=7, label="U-point")
        self.ax.scatter(D_xy[0], D_xy[1], c="red", s=65, edgecolors="white", zorder=7, label="D-point")

        perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

        D_vec = np.array([D_xy[0], D_xy[1]], dtype=float)
        p1_red = D_vec + perp_v * 150
        p2_red = D_vec - perp_v * 150
        self.ax.plot([p1_red[0], p2_red[0]], [p1_red[1], p2_red[1]], color="red", linestyle="-", linewidth=2, zorder=4, label="Apex width line")

        if used_branch_pt is not None:
            B_vec = np.array([used_branch_pt[1], used_branch_pt[0]], dtype=float)
            self.ax.scatter(B_vec[0], B_vec[1], c="lime", s=70, edgecolors="black", marker="s", zorder=7)
            p1_blue = B_vec + perp_v * 150
            p2_blue = B_vec - perp_v * 150
            self.ax.plot([p1_blue[0], p2_blue[0]], [p1_blue[1], p2_blue[1]], color="blue", linestyle="-", linewidth=2, zorder=4, label="Bottom cutoff")

        title_main = f"[{self.index + 1}/{len(self.image_files)}] {search_name}"
        title_sub = " | ".join(result_text)
        self.ax.set_title(f"{title_main}\n{title_sub}", color="white", fontsize=12, pad=14)
        self.ax.axis("off")
        self.ax.legend(fontsize="small", loc="upper right")
        self.fig.tight_layout()

        if reset_zoom or self.current_xlim is None or self.current_ylim is None:
            self.ax.set_xlim(0, img.shape[1])
            self.ax.set_ylim(img.shape[0], 0)
            self.current_xlim = self.ax.get_xlim()
            self.current_ylim = self.ax.get_ylim()
        else:
            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)

        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.image_files)
            self.current_xlim = None
            self.current_ylim = None
            self.update_view(reset_zoom=True)
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.image_files)
            self.current_xlim = None
            self.current_ylim = None
            self.update_view(reset_zoom=True)
        elif event.key == "r":
            self.current_xlim = None
            self.current_ylim = None
            self.update_view(reset_zoom=True)

    def on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1.0

        xdata = event.xdata
        ydata = event.ydata

        cur_width = (cur_xlim[1] - cur_xlim[0])
        cur_height = (cur_ylim[1] - cur_ylim[0])
        new_width = cur_width * scale_factor
        new_height = cur_height * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_width + 1e-12)
        rely = (cur_ylim[1] - ydata) / (cur_height + 1e-12)

        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()
        self.fig.canvas.draw()


"""
===========================================================
10. 실행부
===========================================================
"""
if __name__ == "__main__":
    CSV_PATH = r"capillary_keypoint_final.csv"
    LABEL_DIR = r"D:\usb\MTL_dataset\label"

    if not os.path.exists(CSV_PATH):
        print(f"CSV 파일이 없습니다: {CSV_PATH}")
    elif not os.path.exists(LABEL_DIR):
        print(f"라벨 폴더가 없습니다: {LABEL_DIR}")
    else:
        CapillaryViewer(LABEL_DIR, CSV_PATH)
