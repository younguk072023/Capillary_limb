import cv2
import numpy as np
from collections import deque


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