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


"""
===========================================================
2. skeleton cut 및 내부 hole 처리
===========================================================
"""
def build_cut_skeleton(skeleton, apex_pt, branch_pt=None, dilate_size=3):
    ay, ax = apex_pt
    cut_mask = np.zeros_like(skeleton, dtype=np.uint8)
    cut_mask[ay, ax] = 1

    if branch_pt is not None:
        by, bx = branch_pt
        cut_mask[by, bx] = 1
        cv2.circle(cut_mask, (bx, by), dilate_size, 1, -1)

    if dilate_size > 1:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        cut_mask = cv2.dilate(cut_mask, kernel)

    cut_skeleton = skeleton.copy()
    cut_skeleton[cut_mask > 0] = False
    return cut_skeleton, cut_mask


def find_bottom_of_inner_hole(binary_image, D_pt, dir_v):
    h, w = binary_image.shape
    bg_mask = (~binary_image).astype(np.uint8)

    seed_x, seed_y = None, None
    for k in range(3, 30):
        sx = int(D_pt[0] + dir_v[0] * k)
        sy = int(D_pt[1] + dir_v[1] * k)
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
        proj = np.dot(np.array([hx - D_pt[0], hy - D_pt[1]]), dir_v)
        if proj > max_proj_dist:
            max_proj_dist = proj
            true_bottom_pt = (hy, hx)

    return true_bottom_pt


"""
===========================================================
3. U-D 사이 구간에서 좌/우 limb 시작점 찾기
===========================================================
"""
def sample_perp_line_points(center_xy, perp_v, half_len=20, num=81):
    cx, cy = center_xy
    ts = np.linspace(-half_len, half_len, num)
    pts = []
    for t in ts:
        x = int(round(cx + perp_v[0] * t))
        y = int(round(cy + perp_v[1] * t))
        pts.append((y, x))
    return pts


def find_left_right_seed_on_crossline(skeleton, center_xy, perp_v, img_shape, min_sep=3):
    h, w = img_shape
    pts = sample_perp_line_points(center_xy, perp_v, half_len=25, num=101)

    hits = []
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w and skeleton[y, x]:
            hits.append((y, x))

    if len(hits) < 2:
        return None, None

    xs = np.array([p[1] for p in hits])
    median_x = np.median(xs)

    left_candidates = [p for p in hits if p[1] < median_x]
    right_candidates = [p for p in hits if p[1] >= median_x]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return None, None

    left_seed = min(left_candidates, key=lambda p: p[1])
    right_seed = max(right_candidates, key=lambda p: p[1])

    if abs(right_seed[1] - left_seed[1]) < min_sep:
        return None, None

    return left_seed, right_seed


def find_two_leg_seeds_between_U_and_D(skeleton, U_pt, D_pt):
    Ux, Uy = U_pt
    Dx, Dy = D_pt

    axis_v = np.array([Dx - Ux, Dy - Uy], dtype=float)
    norm = np.linalg.norm(axis_v)
    if norm < 1e-6:
        return None

    dir_v = axis_v / norm
    perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

    candidates = []

    for alpha in np.linspace(0.15, 0.85, 15):
        cx = Ux + alpha * (Dx - Ux)
        cy = Uy + alpha * (Dy - Uy)

        left_seed, right_seed = find_left_right_seed_on_crossline(
            skeleton=skeleton,
            center_xy=(cx, cy),
            perp_v=perp_v,
            img_shape=skeleton.shape,
            min_sep=3
        )

        if left_seed is not None and right_seed is not None:
            score = abs(right_seed[1] - left_seed[1])
            candidates.append((score, left_seed, right_seed))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, left_seed, right_seed = candidates[0]
    return left_seed, right_seed


def trace_leg_from_seed(cut_skeleton, seed_pt):
    dist, parent = geodesic_distances_from_seed(cut_skeleton, seed_pt)
    if len(dist) == 0:
        return []

    end_pt = max(dist.keys(), key=lambda p: dist[p])
    path = reconstruct_path(parent, end_pt)
    return path


def orient_path_from_top_seed(path, U_pt):
    if len(path) < 2:
        return path

    Ux, Uy = U_pt
    d0 = np.hypot(path[0][1] - Ux, path[0][0] - Uy)
    d1 = np.hypot(path[-1][1] - Ux, path[-1][0] - Uy)

    if d1 < d0:
        path = path[::-1]
    return path


def extract_two_leg_paths(skeleton, apex_pt, U_pt, D_pt, branch_pt=None):
    retry_configs = [
        (branch_pt, 3),
        (branch_pt, 1),
        (None, 3),
        (None, 1),
    ]

    best_result = None

    for current_branch, dilate_size in retry_configs:
        cut_skeleton, _ = build_cut_skeleton(skeleton, apex_pt, current_branch, dilate_size)

        seeds = find_two_leg_seeds_between_U_and_D(cut_skeleton, U_pt, D_pt)
        if seeds is None:
            continue

        left_seed, right_seed = seeds

        left_path = trace_leg_from_seed(cut_skeleton, left_seed)
        right_path = trace_leg_from_seed(cut_skeleton, right_seed)

        left_path = orient_path_from_top_seed(left_path, U_pt)
        right_path = orient_path_from_top_seed(right_path, U_pt)

        valid_count = int(len(left_path) >= 5) + int(len(right_path) >= 5)

        result = {
            "paths": [left_path, right_path],
            "branch_pt": current_branch,
            "cut_skeleton": cut_skeleton,
            "left_seed": left_seed,
            "right_seed": right_seed,
            "valid_count": valid_count
        }

        if best_result is None or result["valid_count"] > best_result["valid_count"]:
            best_result = result

        if valid_count == 2:
            return result

    return best_result


"""
===========================================================
4. 순차적 스켈레톤 추적 및 Trim
===========================================================
"""
def trim_path_for_measurement(path, U_pt, D_pt, branch_pt=None, min_keep=8, pixel_margin=1.0):
    vec_axis = np.array([D_pt[0] - U_pt[0], D_pt[1] - U_pt[1]], dtype=float)
    norm = np.linalg.norm(vec_axis)
    dir_v = np.array([0.0, 1.0]) if norm < 1e-5 else vec_axis / norm

    D_vec = np.array([D_pt[0], D_pt[1]], dtype=float)

    if branch_pt is not None:
        B_vec = np.array([branch_pt[1], branch_pt[0]], dtype=float)

    trimmed = []
    recording = False

    for pt in path:
        P = np.array([pt[1], pt[0]], dtype=float)

        proj_D = np.dot(P - D_vec, dir_v)

        if not recording and proj_D >= pixel_margin:
            recording = True

        if recording and branch_pt is not None:
            proj_B = np.dot(P - B_vec, dir_v)
            if proj_B > -pixel_margin:
                break

        if recording:
            trimmed.append(pt)

    if len(trimmed) == 0:
        return [], None, None, dir_v

    if len(trimmed) < min_keep:
        return [], None, None, dir_v

    return trimmed, trimmed[0], trimmed[-1], dir_v


"""
===========================================================
5. stable max diameter
===========================================================
"""
def get_stable_max_diameter(trimmed_path, dist_map,
                            smooth_k=5,
                            top_ratio=0.15,
                            min_plateau_len=3):
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
            "plateau_idx": [best_idx]
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
        "plateau_idx": best_group
    }


"""
===========================================================
6. 단일 이미지 분석 엔진
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

    ux, uy = row["U_x"].values[0], row["U_y"].values[0]
    dx, dy = row["D_x"].values[0], row["D_y"].values[0]

    U_pt = (ux, uy)
    D_pt = (dx, dy)

    vec_axis = np.array([dx - ux, dy - uy], dtype=float)
    norm_axis = np.linalg.norm(vec_axis)
    dir_v = np.array([0.0, 1.0]) if norm_axis < 1e-5 else vec_axis / norm_axis

    ys, xs = np.where(skeleton > 0)
    skeleton_points = np.column_stack((xs, ys))

    u_vec = np.array([ux, uy])
    true_apex_idx = np.argmax(np.linalg.norm(skeleton_points - u_vec, axis=1))
    smx, smy = skeleton_points[true_apex_idx]
    apex_pt = (int(smy), int(smx))

    used_branch_pt = find_bottom_of_inner_hole(binary_image, D_pt, dir_v)

    split_result = extract_two_leg_paths(skeleton, apex_pt, U_pt, D_pt, used_branch_pt)
    if split_result is None:
        return {"ok": False, "reason": "두 다리 path 추출에 실패했습니다."}

    used_branch_pt = split_result["branch_pt"]
    cut_skeleton = split_result["cut_skeleton"]
    raw_paths = split_result["paths"]
    left_seed = split_result.get("left_seed", None)
    right_seed = split_result.get("right_seed", None)

    labeled_cut = label(cut_skeleton)
    _, indices = distance_transform_edt(labeled_cut == 0, return_indices=True)
    labeled_mask = labeled_cut[indices[0], indices[1]]
    labeled_mask[~binary_image] = 0

    dist_map = distance_transform_edt(binary_image)

    leg_results = []

    for idx, path in enumerate(raw_paths):
        if len(path) < 2:
            continue

        trimmed_path, start_pt, end_pt, _ = trim_path_for_measurement(
            path,
            U_pt,
            D_pt,
            used_branch_pt,
            min_keep=8,
            pixel_margin=1.0
        )

        if len(trimmed_path) == 0:
            continue

        stable_res = get_stable_max_diameter(
            trimmed_path,
            dist_map,
            smooth_k=5,
            top_ratio=0.15,
            min_plateau_len=3
        )

        if stable_res is None:
            continue

        trimmed_xs = np.array([p[1] for p in trimmed_path])

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
            "plateau_idx": stable_res["plateau_idx"]
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
        "U_pt": U_pt,
        "D_pt": D_pt,
        "used_branch_pt": used_branch_pt,
        "dir_v": dir_v,
        "left_seed": left_seed,
        "right_seed": right_seed
    }


"""
===========================================================
7. 키보드/휠 인터랙티브 뷰어
===========================================================
"""
class CapillaryViewer:
    def __init__(self, image_dir, csv_path):
        self.image_dir = image_dir
        self.csv_path = csv_path

        self.df_keypoints = pd.read_csv(csv_path)

        valid_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]
        )

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_dir}")

        self.index = 0
        self.current_xlim = None
        self.current_ylim = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor="black")
        self.fig.canvas.manager.set_window_title("Capillary Viewer")

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
                0.5, 0.5,
                f"[{self.index + 1}/{len(self.image_files)}]\n{os.path.basename(image_path)}\n\n{result['reason']}",
                transform=self.ax.transAxes,
                ha="center", va="center",
                color="red", fontsize=12
            )
            self.ax.axis("off")
            self.fig.tight_layout()
            self.fig.canvas.draw()
            return

        img = result["img"]
        leg_results = result["leg_results"]
        labeled_mask = result["labeled_mask"]
        U_pt = result["U_pt"]
        D_pt = result["D_pt"]
        used_branch_pt = result["used_branch_pt"]
        dir_v = result["dir_v"]
        left_seed = result["left_seed"]
        right_seed = result["right_seed"]
        search_name = result["search_name"]

        self.ax.imshow(img, cmap="gray")

        # 배경 위에 labeled mask 살짝 덮기
        overlay = np.ma.masked_where(labeled_mask == 0, labeled_mask)
        self.ax.imshow(overlay, cmap="nipy_spectral", alpha=0.25)

        # 결과 텍스트
        result_text = []
        if len(leg_results) >= 2:
            sorted_leg_results = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)
            result_text.append(f"Venous: {sorted_leg_results[0]['max_d']:.2f}px")
            result_text.append(f"Arterial: {sorted_leg_results[1]['max_d']:.2f}px")
        else:
            for i, r in enumerate(leg_results):
                result_text.append(f"Leg{i+1}: {r['max_d']:.2f}px")

        # path 표시
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

            self.ax.scatter(
                res["max_x"], res["max_y"],
                c="red", s=70, edgecolors="white", zorder=6
            )
            self.ax.text(
                res["max_x"], res["max_y"] - 12,
                f"{res['max_d']:.2f}",
                color="white", fontsize=9, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6)
            )

        if left_seed is not None:
            self.ax.scatter(left_seed[1], left_seed[0], c="cyan", s=70, edgecolors="black", marker="o", zorder=7)
        if right_seed is not None:
            self.ax.scatter(right_seed[1], right_seed[0], c="yellow", s=70, edgecolors="black", marker="o", zorder=7)

        # U, D
        self.ax.scatter(U_pt[0], U_pt[1], c="gray", s=55, edgecolors="white", zorder=7, label="U-point")
        self.ax.scatter(D_pt[0], D_pt[1], c="red", s=65, edgecolors="white", zorder=7, label="D-point")

        perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

        # red line
        D_vec = np.array([D_pt[0], D_pt[1]], dtype=float)
        p1_red = D_vec + perp_v * 150
        p2_red = D_vec - perp_v * 150
        self.ax.plot(
            [p1_red[0], p2_red[0]],
            [p1_red[1], p2_red[1]],
            color="red", linestyle="-", linewidth=2,
            zorder=4, label="Apex width line"
        )

        # blue line
        if used_branch_pt is not None:
            B_vec = np.array([used_branch_pt[1], used_branch_pt[0]], dtype=float)
            self.ax.scatter(B_vec[0], B_vec[1], c="lime", s=70, edgecolors="black", marker="s", zorder=7)

            p1_blue = B_vec + perp_v * 150
            p2_blue = B_vec - perp_v * 150
            self.ax.plot(
                [p1_blue[0], p2_blue[0]],
                [p1_blue[1], p2_blue[1]],
                color="blue", linestyle="-", linewidth=2,
                zorder=4, label="Bottom cutoff"
            )

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
8. 실행부
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