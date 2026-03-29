import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from collections import deque
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


def bfs_shortest_path_within_mask(comp_mask, start, goal):
    q = deque([start])
    parent = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break

        for nxt in get_neighbors(cur[0], cur[1], comp_mask):
            if nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    if goal not in parent:
        return [start]

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


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
    """
    D점 아래쪽 내부 hole에 seed를 찍고,
    그 connected component 중 dir_v 방향으로 가장 아래에 있는 점을 반환
    반환: (y, x) or None
    """
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
3. 두 다리 seed를 apex 근처에서 직접 찾기
===========================================================
"""
def get_candidate_seeds_near_apex(cut_skeleton, apex_pt, max_radius=12):
    """
    apex 주변에서 서로 다른 방향으로 나가는 skeleton seed 후보를 찾음
    """
    ay, ax = apex_pt
    h, w = cut_skeleton.shape
    candidates = []

    for r in range(1, max_radius + 1):
        for y in range(max(0, ay - r), min(h, ay + r + 1)):
            for x in range(max(0, ax - r), min(w, ax + r + 1)):
                if not cut_skeleton[y, x]:
                    continue
                d = np.hypot(y - ay, x - ax)
                if d <= r:
                    candidates.append((y, x))

        if len(candidates) >= 2:
            break

    # 중복/가까운 점 제거
    unique = []
    for pt in sorted(candidates, key=lambda p: np.hypot(p[0]-ay, p[1]-ax)):
        too_close = False
        for q in unique:
            if np.hypot(pt[0]-q[0], pt[1]-q[1]) <= 2:
                too_close = True
                break
        if not too_close:
            unique.append(pt)

    return unique


def choose_two_leg_seeds(cut_skeleton, apex_pt):
    """
    apex 주변 seed 후보 중 좌우로 가장 잘 벌어지는 2개를 선택
    """
    ay, ax = apex_pt
    candidates = get_candidate_seeds_near_apex(cut_skeleton, apex_pt, max_radius=15)

    if len(candidates) < 2:
        ys, xs = np.where(cut_skeleton)
        pts = list(zip(ys, xs))
        if len(pts) < 2:
            return []
        pts = sorted(pts, key=lambda p: np.hypot(p[0]-ay, p[1]-ax))
        candidates = pts[:20]

    best_pair = None
    best_score = -float("inf")

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            p1, p2 = candidates[i], candidates[j]

            v1 = np.array([p1[1] - ax, p1[0] - ay], dtype=float)
            v2 = np.array([p2[1] - ax, p2[0] - ay], dtype=float)

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue

            # 각도가 충분히 벌어지고, 좌우 분리가 되는 pair 선호
            cosang = np.dot(v1, v2) / (n1 * n2)
            sep_x = abs(p1[1] - p2[1])
            score = (1.0 - cosang) * 10.0 + sep_x

            if score > best_score:
                best_score = score
                best_pair = [p1, p2]

    if best_pair is None:
        return []

    best_pair = sorted(best_pair, key=lambda p: p[1])  # left, right
    return best_pair


def trace_single_leg_path(cut_skeleton, seed_pt):
    """
    seed에서 가장 멀리 있는 skeleton 점까지 path 추적
    """
    dist, parent = geodesic_distances_from_seed(cut_skeleton, seed_pt)
    if len(dist) == 0:
        return []

    end_pt = max(dist.keys(), key=lambda p: dist[p])
    path = reconstruct_path(parent, end_pt)
    return path


def extract_two_leg_paths(skeleton, apex_pt, branch_pt=None):
    """
    핵심:
    component selected_ids에 의존하지 않고,
    apex cut 이후 seed 2개를 직접 찾아 각각 path를 뽑음
    """
    retry_configs = [
        (branch_pt, 3),
        (branch_pt, 1),
        (None, 3),
        (None, 1),
    ]

    best_paths = []

    for current_branch, dilate_size in retry_configs:
        cut_skeleton, _ = build_cut_skeleton(skeleton, apex_pt, current_branch, dilate_size)
        seeds = choose_two_leg_seeds(cut_skeleton, apex_pt)

        if len(seeds) < 2:
            continue

        paths = []
        for seed in seeds:
            path = trace_single_leg_path(cut_skeleton, seed)
            if len(path) >= 5:
                paths.append(path)

        if len(paths) > len(best_paths):
            best_paths = paths

        if len(paths) == 2:
            return {
                "paths": sorted(paths, key=lambda path: np.mean([p[1] for p in path])),
                "branch_pt": current_branch,
                "cut_skeleton": cut_skeleton
            }

    if len(best_paths) > 0:
        return {
            "paths": sorted(best_paths, key=lambda path: np.mean([p[1] for p in path])),
            "branch_pt": branch_pt,
            "cut_skeleton": cut_skeleton
        }

    return None


"""
===========================================================
4. path trim
===========================================================
"""
def trim_path_between_red_and_blue_lines(path, U_pt, D_pt, branch_pt=None,
                                         end_margin_ratio=0, min_keep=8):
    vec_axis = np.array([D_pt[0] - U_pt[0], D_pt[1] - U_pt[1]], dtype=float)
    norm = np.linalg.norm(vec_axis)
    dir_v = np.array([0.0, 1.0]) if norm < 1e-5 else vec_axis / norm

    D_vec = np.array([D_pt[0], D_pt[1]], dtype=float)

    if branch_pt is not None:
        B_vec = np.array([branch_pt[1], branch_pt[0]], dtype=float)

    trimmed = []
    for pt in path:
        P = np.array([pt[1], pt[0]], dtype=float)

        proj_D = np.dot(P - D_vec, dir_v)
        if proj_D < 0:
            continue

        if branch_pt is not None:
            proj_B = np.dot(P - B_vec, dir_v)
            if proj_B > 0:
                continue

        trimmed.append(pt)

    if len(trimmed) == 0:
        return [], None, None, dir_v

    if len(trimmed) < min_keep:
        return trimmed, trimmed[0], trimmed[-1], dir_v

    n = len(trimmed)
    margin = max(1, int(n * end_margin_ratio))

    if n - 2 * margin >= min_keep:
        trimmed = trimmed[margin:n - margin]

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
6. 시각화 및 출력
===========================================================
"""
def plot_and_print_results(search_name, leg_results, labeled_mask, U_pt, D_pt, used_branch_pt, dir_v):
    print(f"\n{'=' * 60}")
    print(f"이미지 파일: {search_name}")

    # 두 다리 다 구한 뒤 큰 쪽=세정맥, 다음=세동맥
    sorted_leg_results = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)

    if len(sorted_leg_results) >= 2:
        sorted_leg_results[0]["vessel_type"] = "Venous"
        sorted_leg_results[1]["vessel_type"] = "Arterial"

        print(f"- 세정맥(Venous)")
        print(f"  · 최대 limb 직경: {sorted_leg_results[0]['max_d']:.2f} px")
        print(f"- 세동맥(Arterial)")
        print(f"  · 최대 limb 직경: {sorted_leg_results[1]['max_d']:.2f} px")
    else:
        for idx, res in enumerate(sorted_leg_results):
            print(f"- Leg {idx + 1}")
            print(f"  · 최대 limb 직경: {res['max_d']:.2f} px")

    print(f"{'=' * 60}\n")

    plt.figure(figsize=(8, 8), facecolor="black")
    plt.imshow(labeled_mask, cmap="nipy_spectral")

    # 좌우 정렬 기준 시각화
    for idx, res in enumerate(leg_results):
        side = "Left Leg" if idx == 0 else "Right Leg"

        px = [p[1] for p in res["path"]]
        py = [p[0] for p in res["path"]]
        plt.plot(px, py, color="white", linewidth=1.2, alpha=0.45, zorder=1)

        tx = [p[1] for p in res["trimmed_path"]]
        ty = [p[0] for p in res["trimmed_path"]]
        plt.plot(tx, ty, color="black", linewidth=3, label=f"{side} trimmed" if idx == 0 else None, zorder=2)

        label_txt = res.get("final_label", side)
        plt.scatter(
            res["max_x"], res["max_y"],
            c="red", s=85, edgecolors="white", zorder=6,
            label=f"{label_txt} max" if idx == 0 else None
        )

    D_vec = np.array([D_pt[0], D_pt[1]], dtype=float)

    plt.scatter(U_pt[0], U_pt[1], c="gray", s=55, edgecolors="white", label="U-point", zorder=5)
    plt.scatter(D_vec[0], D_vec[1], c="red", s=65, edgecolors="white", label="D-point", zorder=5)

    perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

    p1_red = D_vec + perp_v * 150
    p2_red = D_vec - perp_v * 150
    plt.plot(
        [p1_red[0], p2_red[0]],
        [p1_red[1], p2_red[1]],
        color="red", linestyle="-", linewidth=2,
        label="Top cutoff", zorder=3
    )

    if used_branch_pt is not None:
        B_vec = np.array([used_branch_pt[1], used_branch_pt[0]], dtype=float)

        plt.scatter(
            B_vec[0], B_vec[1],
            c="lime", s=70, edgecolors="black", marker="s",
            label="Inner-hole bottom", zorder=5
        )

        p1_blue = B_vec + perp_v * 150
        p2_blue = B_vec - perp_v * 150
        plt.plot(
            [p1_blue[0], p2_blue[0]],
            [p1_blue[1], p2_blue[1]],
            color="blue", linestyle="-", linewidth=2,
            label="Bottom cutoff", zorder=3
        )

    plt.title("Two-Leg Limb Diameter Measurement", color="white")
    plt.legend(fontsize="small", loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


"""
===========================================================
7. 메인
===========================================================
"""
def skeletonize_image(image_path, csv_file):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("이미지를 읽지 못했습니다.")
        return

    binary_image = img > 127
    skeleton = skeletonize(binary_image)

    labeled_full = label(skeleton)
    if labeled_full.max() == 0:
        print("스켈레톤이 없습니다.")
        return

    counts = np.bincount(labeled_full.flat)
    main_label = np.argmax(counts[1:]) + 1
    skeleton = (labeled_full == main_label)

    df = pd.read_csv(csv_file)
    search_name = os.path.splitext(os.path.basename(image_path))[0]
    row = df[df["filename"].apply(lambda x: os.path.splitext(str(x))[0]) == search_name]

    if row.empty:
        print(f"{search_name} 에 해당하는 CSV 행이 없습니다.")
        return

    ux, uy = row["U_x"].values[0], row["U_y"].values[0]
    dx, dy = row["D_x"].values[0], row["D_y"].values[0]

    # 원본 코드 좌표계 유지
    U_pt = (ux, uy)
    D_pt = (dx, dy)

    vec_axis = np.array([dx - ux, dy - uy], dtype=float)
    norm_axis = np.linalg.norm(vec_axis)
    dir_v = np.array([0.0, 1.0]) if norm_axis < 1e-5 else vec_axis / norm_axis

    ys, xs = np.where(skeleton > 0)
    skeleton_points = np.column_stack((xs, ys))

    apex_pt_idx = np.argmin(np.linalg.norm(skeleton_points - np.array([dx, dy]), axis=1))
    smx, smy = skeleton_points[apex_pt_idx]
    apex_pt = (int(smy), int(smx))   # (y, x)

    used_branch_pt = find_bottom_of_inner_hole(binary_image, D_pt, dir_v)

    # 핵심 변경:
    # component id 기반이 아니라 seed 2개 기반으로 두 path를 직접 확보
    split_result = extract_two_leg_paths(skeleton, apex_pt, used_branch_pt)
    if split_result is None or len(split_result["paths"]) == 0:
        print("두 다리 path 추출에 실패했습니다.")
        return

    used_branch_pt = split_result["branch_pt"]
    cut_skeleton = split_result["cut_skeleton"]
    raw_paths = split_result["paths"]

    # 보기용 labeled mask
    labeled_cut = label(cut_skeleton)
    _, indices = distance_transform_edt(labeled_cut == 0, return_indices=True)
    labeled_mask = labeled_cut[indices[0], indices[1]]
    labeled_mask[~binary_image] = 0

    dist_map = distance_transform_edt(binary_image)

    leg_results = []

    for idx, path in enumerate(raw_paths):
        if len(path) < 2:
            continue

        trimmed_path, start_pt, end_pt, _ = trim_path_between_red_and_blue_lines(
            path,
            U_pt,
            D_pt,
            used_branch_pt,
            end_margin_ratio=0,
            min_keep=8
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
        print("유효한 limb 결과가 없습니다.")
        return

    # 좌우 정렬
    leg_results.sort(key=lambda x: x["mean_x"])

    # 최종 라벨링: 두 개 다 구한 뒤 더 큰 쪽을 Venous, 나머지를 Arterial
    if len(leg_results) >= 2:
        by_size = sorted(range(len(leg_results)), key=lambda i: leg_results[i]["max_d"], reverse=True)
        leg_results[by_size[0]]["final_label"] = "Venous"
        leg_results[by_size[1]]["final_label"] = "Arterial"

    plot_and_print_results(
        search_name=search_name,
        leg_results=leg_results,
        labeled_mask=labeled_mask,
        U_pt=U_pt,
        D_pt=D_pt,
        used_branch_pt=used_branch_pt,
        dir_v=dir_v
    )


if __name__ == "__main__":
    CSV_FILE = "capillary_keypoint_final.csv"
    IMAGE_FILE = "p1_det_050.tif"
    skeletonize_image(IMAGE_FILE, CSV_FILE)