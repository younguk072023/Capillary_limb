'''
곡률 계산 및 경로 추적 알고리즘

'''
import numpy as np
from scipy.signal import savgol_filter
from image_utils import get_neighbors, find_endpoints

def order_component_path(comp_mask, apex_ref, branch_ref=None):
    ys, xs = np.where(comp_mask)
    if len(ys) == 0: return []

    points = list(zip(ys, xs))
    endpoints = find_endpoints(comp_mask)

    #stary와 가까운 점
    if len(endpoints) >= 1:
        start_pt = min(endpoints, key=lambda p: np.hypot(p[1]-apex_ref[1], p[0]-apex_ref[0]))
    else:
        start_pt = min(points, key=lambda p: np.hypot(p[1]-apex_ref[1], p[0]-apex_ref[0]))

    end_pt = None
    #branch와 가까운 점
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

        nxt = candidates[0] if len(candidates) == 1 else (
            min(candidates, key=lambda p: np.hypot(p[1]-end_pt[1], p[0]-end_pt[0])) if end_pt else candidates[0]
        )

        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

        if end_pt is not None and cur == end_pt:
            break

    return path

#굽어있는 꼭대기와 갈림길 제외 두께가 가장 잘 나와있는 다리 부분
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

    #급커브 점수 매기기
    dx = np.gradient(xs_smooth)
    dy = np.gradient(ys_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    #곡률 공식
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-8
    curvature = np.abs(dx * ddy - dy * ddx) / denominator

    #주관적 개입 -> 논문 search
    top_search_range = min(n, max(15, n // 3))
    apex_peak_idx = np.argmax(curvature[:top_search_range])
    apex_max_k = curvature[apex_peak_idx]
    
    start_idx = apex_peak_idx
    for i in range(apex_peak_idx, n):
        if curvature[i] < apex_max_k * 0.20:
            start_idx = i
            break

    bottom_search_start = max(0, n - max(15, n // 3))
    branch_peak_idx = bottom_search_start + np.argmax(curvature[bottom_search_start:])
    branch_max_k = curvature[branch_peak_idx]
    
    end_idx = branch_peak_idx
    for i in range(branch_peak_idx, -1, -1):
        if curvature[i] < branch_max_k * 0.20:
            end_idx = i
            break

    if start_idx >= end_idx:
        start_idx = n // 4
        end_idx = n - (n // 4)

    trimmed_path = path[start_idx:end_idx+1]
    return trimmed_path, path[start_idx], path[end_idx]