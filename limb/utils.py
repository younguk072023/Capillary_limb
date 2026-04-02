import cv2
import numpy as np
from collections import deque

# BFS, 혈관 뼈대 위를 따라가며 왼쪽 다리, 오른쪽 다리 경로를 추적할 때 쓰이는 길찾기 알고리즘
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

# 평균내어 수평내는 곳 smootk_k
def smooth_1d(arr, k=5):
    if len(arr) < k or k <= 1:
        return arr.copy()
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(padded, kernel, mode="valid")

# 시작점 seed를 통한 출발 점 찾기
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

#BFS로부터 부모노드 역추적
def reconstruct_path(parent, end_pt):
    path = []
    cur = end_pt
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

# 한글 파일명에 대한 이미지 읽기 함수
def read_image_unicode(image_path):
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return img

#좌표를 Skeleton에 붙이는 부분
def snap_xy_to_skeleton(skeleton, pt_xy):
    xs = np.where(skeleton)[1]
    ys = np.where(skeleton)[0]
    if len(xs) == 0:
        return None

    px, py = pt_xy
    d2 = (xs - px) ** 2 + (ys - py) ** 2
    idx = int(np.argmin(d2))
    return int(ys[idx]), int(xs[idx])

# 미세한 노이즈 분할된 영역 제거하고 가장 큰 부분의 혈관 덩어리만 남기는 함수
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

# 혈관의 끝점 찾기
def find_endpoints(mask):
    ys, xs = np.where(mask)
    endpoints = []
    for y, x in zip(ys, xs):
        deg = len(get_neighbors(y, x, mask))
        if deg <= 1:
            endpoints.append((y, x))
    return endpoints