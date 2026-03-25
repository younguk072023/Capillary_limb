import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve, distance_transform_edt
from scipy.signal import savgol_filter

# ============================================================
# 8-이웃 검색 및 끝점 찾기
# ============================================================
def get_neighbors(y, x, mask):
    neighbors = []
    h, w = mask.shape
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors

def find_endpoints(comp_mask):
    ys, xs = np.where(comp_mask)
    endpoints = []
    for y, x in zip(ys, xs):
        if len(get_neighbors(y, x, comp_mask)) == 1:
            endpoints.append((y, x))
    return endpoints

# ============================================================
# 스켈레톤 조각을 Apex쪽 -> Branch쪽 '선(Path)'으로 정렬
# ============================================================
def order_component_path(comp_mask, apex_ref, branch_ref=None):
    ys, xs = np.where(comp_mask)
    if len(ys) == 0: return []

    points = list(zip(ys, xs))
    endpoints = find_endpoints(comp_mask)

    if len(endpoints) >= 1:
        start_pt = min(endpoints, key=lambda p: np.hypot(p[1]-apex_ref[1], p[0]-apex_ref[0]))
    else:
        start_pt = min(points, key=lambda p: np.hypot(p[1]-apex_ref[1], p[0]-apex_ref[0]))

    end_pt = None
    if len(endpoints) >= 2:
        candidates = [p for p in endpoints if p != start_pt]
        if branch_ref is not None:
            end_pt = min(candidates, key=lambda p: np.hypot(p[1]-branch_ref[1], p[0]-branch_ref[0]))
        else:
            end_pt = max(candidates, key=lambda p: np.hypot(p[1]-start_pt[1], p[0]-start_pt[0]))

    path = [start_pt]
    visited = {start_pt}
    prev = None
    cur = start_pt

    while True:
        neighs = [p for p in get_neighbors(cur[0], cur[1], comp_mask) if p != prev]
        if len(neighs) == 0: break

        candidates = [p for p in neighs if p not in visited]
        if len(candidates) == 0: break

        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            if end_pt is not None:
                nxt = min(candidates, key=lambda p: np.hypot(p[1]-end_pt[1], p[0]-end_pt[0]))
            else:
                nxt = candidates[0]

        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

        if end_pt is not None and cur == end_pt:
            break

    return path

# ============================================================
# 💡 핵심: 양 끝단(Apex & Branch)의 곡률을 모두 계산해 진짜 직선 구간 찾기
# ============================================================
def find_limb_valid_range_by_curvature(path):
    n = len(path)
    if n < 10:
        return path, path[0], path[-1]

    window_length = min(15, n)
    if window_length % 2 == 0: window_length -= 1
    if window_length < 3: window_length = 3

    xs = np.array([p[1] for p in path], dtype=float)
    ys = np.array([p[0] for p in path], dtype=float)

    xs_smooth = savgol_filter(xs, window_length, 2)
    ys_smooth = savgol_filter(ys, window_length, 2)

    dx = np.gradient(xs_smooth)
    dy = np.gradient(ys_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-8
    curvature = np.abs(dx * ddy - dy * ddx) / denominator

    # 1. 위쪽(Apex) 굴곡 배제: 시작점에서부터 내려가며 곡률 하강점 찾기
    top_search_range = min(n, max(15, n // 3))
    apex_peak_idx = np.argmax(curvature[:top_search_range])
    apex_max_k = curvature[apex_peak_idx]
    
    start_idx = apex_peak_idx
    for i in range(apex_peak_idx, n):
        if curvature[i] < apex_max_k * 0.20:
            start_idx = i
            break

    # 2. 아래쪽(Branch) 굴곡 배제: 끝점에서부터 거슬러 올라가며 곡률 하강점 찾기
    bottom_search_start = max(0, n - max(15, n // 3))
    branch_peak_idx = bottom_search_start + np.argmax(curvature[bottom_search_start:])
    branch_max_k = curvature[branch_peak_idx]
    
    end_idx = branch_peak_idx
    for i in range(branch_peak_idx, -1, -1):
        if curvature[i] < branch_max_k * 0.20:
            end_idx = i
            break

    # 안전 장치: 시작점이 끝점보다 역전되거나 겹칠 경우 (다리 전체가 그냥 둥근 경우)
    if start_idx >= end_idx:
        start_idx = n // 4
        end_idx = n - (n // 4)

    trimmed_path = path[start_idx:end_idx+1]
    return trimmed_path, path[start_idx], path[end_idx]

# ============================================================
# Branch 탐지 및 두 다리 분리 로직 
# ============================================================
def get_branch_candidate(skeleton, apex_pt):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    filtered = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    branch_ys, branch_xs = np.where(filtered >= 13)

    if len(branch_xs) == 0:
        return None, branch_xs, branch_ys

    ax, ay = apex_pt[1], apex_pt[0]
    branch_points = np.column_stack((branch_xs, branch_ys))
    dist = np.linalg.norm(branch_points - np.array([ax, ay]), axis=1)

    valid = np.where(dist >= 3.0)[0]
    idx = valid[np.argmin(dist[valid])] if len(valid) > 0 else np.argmin(dist)
    bx, by = branch_points[idx]
    return (by, bx), branch_xs, branch_ys

def build_cut_skeleton(skeleton, apex_pt, branch_pt=None, dilate_size=3):
    ay, ax = apex_pt
    cut_mask = np.zeros_like(skeleton, dtype=np.uint8)
    cut_mask[ay, ax] = 1

    if branch_pt is not None:
        by, bx = branch_pt
        cut_mask[by, bx] = 1
        cv2.circle(cut_mask, (bx, by), dilate_size, 1, -1)

    if dilate_size > 1:
        cut_mask = cv2.dilate(cut_mask, np.ones((dilate_size, dilate_size), np.uint8))

    cut_skeleton = skeleton.copy()
    cut_skeleton[cut_mask > 0] = False
    return cut_skeleton, cut_mask

def split_two_legs_with_retry(skeleton, apex_pt, branch_pt):
    retry_configs = [(branch_pt, 3), (branch_pt, 1), (None, 3), (None, 1)]
    best = None

    for current_branch, dilate_size in retry_configs:
        cut_skeleton, cut_mask = build_cut_skeleton(skeleton, apex_pt, current_branch, dilate_size)
        labeled_skel = label(cut_skeleton)
        
        comps = []
        for i in range(1, labeled_skel.max() + 1):
            ys, xs = np.where(labeled_skel == i)
            if len(ys) >= 8:
                dist_to_apex = np.min(np.hypot(xs - apex_pt[1], ys - apex_pt[0]))
                comps.append({'id': i, 'mean_x': np.mean(xs), 'dist': dist_to_apex})

        if len(comps) >= 2:
            comps.sort(key=lambda d: d["dist"])
            best_two = comps[:2]
            best_two.sort(key=lambda d: d["mean_x"]) 
            selected_ids = [c['id'] for c in best_two]
        else:
            selected_ids = [c['id'] for c in comps]

        candidate = {
            "labeled_skel": labeled_skel,
            "selected_ids": selected_ids,
            "branch_pt": current_branch
        }

        if best is None or len(selected_ids) > len(best.get("selected_ids", [])):
            best = candidate
        if len(selected_ids) == 2:
            return candidate
            
    return best

# ============================================================
# 메인 함수
# ============================================================
def skeletonize_image(image, CSV_FILE):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    binary_image = img > 127
    skeleton = skeletonize(binary_image)

    labeled_full = label(skeleton)
    skeleton = (labeled_full == np.argmax(np.bincount(labeled_full.flat)[1:]) + 1)

    df = pd.read_csv(CSV_FILE)
    search_name = os.path.splitext(os.path.basename(image))[0]
    row = df[df["filename"].apply(lambda x: os.path.splitext(str(x))[0]) == search_name]
    if row.empty: return

    ux, uy = row["U_x"].values[0], row["U_y"].values[0]
    dx, dy = row["D_x"].values[0], row["D_y"].values[0]

    ys, xs = np.where(skeleton > 0)
    skeleton_points = np.column_stack((xs, ys))
    mid_x, mid_y = (ux + dx) / 2, (uy + dy) / 2

    smx, smy = skeleton_points[np.argmin(np.linalg.norm(skeleton_points - np.array([mid_x, mid_y]), axis=1))]
    apex_pt = (int(smy), int(smx))

    branch_pt, branch_xs, branch_ys = get_branch_candidate(skeleton, apex_pt)
    split_result = split_two_legs_with_retry(skeleton, apex_pt, branch_pt)
    labeled_skel = split_result["labeled_skel"]
    selected_ids = split_result["selected_ids"]
    used_branch_pt = split_result["branch_pt"]

    _, indices = distance_transform_edt(labeled_skel == 0, return_indices=True)
    labeled_mask = labeled_skel[indices[0], indices[1]]
    labeled_mask[~binary_image] = 0
    dist_map = distance_transform_edt(binary_image)

    leg_results = []

    for comp_id in selected_ids:
        path = order_component_path((labeled_skel == comp_id), apex_ref=apex_pt, branch_ref=used_branch_pt)
        if len(path) < 2: continue

        # 💡 양방향 곡률을 모두 분석하여 정확한 중앙 다리 구간 추출
        trimmed_path, start_pt, end_pt = find_limb_valid_range_by_curvature(path)

        if len(trimmed_path) == 0: continue

        trimmed_ys = np.array([p[0] for p in trimmed_path])
        trimmed_xs = np.array([p[1] for p in trimmed_path])

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

    # --------------------------------------------------------
    # 출력 및 시각화
    # --------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"이미지 파일: {search_name}")
    print("측정 로직: 양 끝단(Apex & Branch) 굴곡 배제 (곡률 기반)")
    
    for idx, res in enumerate(leg_results):
        side = "세동맥(Arterial)" if idx == 0 else "세정맥(Venous)"
        print(f"- {side}")
        print(f"  · 측정 구간: ({res['start_pt'][1]}, {res['start_pt'][0]}) ~ ({res['end_pt'][1]}, {res['end_pt'][0]})")
        print(f"  · 최대 직경: {res['max_d']:.2f} px")
    print(f"{'='*50}\n")

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 4)
    plt.imshow(labeled_mask, cmap="nipy_spectral")

    colors = ["cyan", "lime"]
    for idx, res in enumerate(leg_results):
        side = "Arterial" if idx == 0 else "Venous"
        color = colors[idx % 2]

        # 1. 잘려나간 전체 경로 (회색 실선)
        px, py = [p[1] for p in res["path"]], [p[0] for p in res["path"]]
        plt.plot(px, py, color="lightgray", linewidth=2, zorder=1)

        # 2. 곡률 필터를 통과한 진짜 측정 구간 (색상 실선)
        tx, ty = [p[1] for p in res["trimmed_path"]], [p[0] for p in res["trimmed_path"]]
        plt.plot(tx, ty, color=color, linewidth=2, label=f"{side} Range", zorder=2)

        # 3. 측정 시작점 (Apex쪽)과 끝점 (Branch쪽) (노란색)
        plt.scatter(res["start_pt"][1], res["start_pt"][0], c="yellow", s=60, edgecolors="black", label=f"{side} Start", zorder=3)
        plt.scatter(res["end_pt"][1], res["end_pt"][0], c="yellow", s=60, edgecolors="black", marker='s', label=f"{side} End", zorder=3)

        # 4. 최대 직경 포인트 (빨간 별)
        plt.scatter(res["max_x"], res["max_y"], c="red", s=150, marker="*", edgecolors="white", zorder=4)

    plt.scatter(apex_pt[1], apex_pt[0], c="white", s=60, edgecolors="black", label="Apex", zorder=5)
    if used_branch_pt:
        plt.scatter(used_branch_pt[1], used_branch_pt[0], c="white", marker="X", s=80, edgecolors="black", label="Branch", zorder=5)

    plt.title("Double Curvature Trimming (Apex & Branch)")
    plt.legend(fontsize='small', loc="upper right")
    plt.tight_layout()
    plt.show()

CSV_FILE = "capillary_keypoint_final.csv"
skeletonize_image("p9_det_019_crop_3.tif", CSV_FILE)