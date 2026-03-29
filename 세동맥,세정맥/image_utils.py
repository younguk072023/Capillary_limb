'''
이미지 처리 및 스켈레톤 기본 tool
1. skeleton 이미지에서 이웃 픽셀 찾음.
2. 끝점 찾고 branch 후보 찾음
3. apex/branch를 잘라냄.
4. 두 다리로 분리함.
'''

import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage.measure import label

# 이웃 픽셀 찾기
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

# skeleton안에서 끝점 찾는 함수
def find_endpoints(comp_mask):
    ys, xs = np.where(comp_mask)
    endpoints = []
    for y, x in zip(ys, xs):
        if len(get_neighbors(y, x, comp_mask)) == 1:
            endpoints.append((y, x))
    return endpoints

# branch 후보 찾고, apex거리 후보에 있어서 가장 가까운 branch점 찾는 함수
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

# apex와 branch 주변 픽셀을 일부러 지워서 이어진 스켈레톤을 끊어 놓는 함수
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

# skeleton에서 apex/branch를 잘라내고, 두 다리로 분리하는 함수
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