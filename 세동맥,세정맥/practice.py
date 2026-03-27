import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
import random

# ==========================================
# 1. 영욱님의 핵심 알고리즘 (기존 로직)
# ==========================================

def get_neighbors(y, x, mask):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                if mask[ny, nx] > 0:
                    neighbors.append((ny, nx))
    return neighbors

def find_endpoints(mask):
    ys, xs = np.where(mask)
    endpoints = []
    for y, x in zip(ys, xs):
        if len(get_neighbors(y, x, mask)) <= 1:
            endpoints.append((y, x))
    return endpoints

def order_component_path(comp_mask, apex_ref):
    ys, xs = np.where(comp_mask)
    if len(ys) == 0: return []
    endpoints = find_endpoints(comp_mask)
    
    # 꼭대기(Apex)와 가장 가까운 끝점을 시작점으로 설정
    if len(endpoints) >= 1:
        start_pt = min(endpoints, key=lambda p: np.hypot(p[1]-apex_ref[1], p[0]-apex_ref[0]))
    else:
        points = list(zip(ys, xs))
        start_pt = min(points, key=lambda p: np.hypot(p[1]-apex_ref[1], p[0]-apex_ref[0]))

    path, visited = [start_pt], {start_pt}
    prev, cur = None, start_pt

    while True:
        neighs = [p for p in get_neighbors(cur[0], cur[1], comp_mask) if p != prev]
        candidates = [p for p in neighs if p not in visited]
        if not candidates: break
        
        # 다음 점으로 이동
        nxt = candidates[0]
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
    return path

def find_limb_valid_range_by_curvature(path):
    n = len(path)
    if n < 15: return path, path[0], path[-1]
    
    # Savitzky-Golay 필터로 매끄럽게 만들기
    window_length = min(15, n)
    if window_length % 2 == 0: window_length -= 1
    
    xs = np.array([p[1] for p in path], dtype=float)
    ys = np.array([p[0] for p in path], dtype=float)
    xs_s = savgol_filter(xs, window_length, 2)
    ys_s = savgol_filter(ys, window_length, 2)

    # 곡률(Curvature) 계산
    dx, dy = np.gradient(xs_s), np.gradient(ys_s)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    den = (dx**2 + dy**2)**1.5
    den[den == 0] = 1e-8
    curv = np.abs(dx * ddy - dy * ddx) / den

    # 꼭대기에서 20% 감쇄 지점 찾기 (영욱님의 핵심 기준)
    search_limit = min(n, max(15, n // 3))
    apex_idx = np.argmax(curv[:search_limit])
    apex_k = curv[apex_idx]
    
    start_idx = apex_idx
    for i in range(apex_idx, n):
        if curv[i] < apex_k * 0.20: # 20% Threshold
            start_idx = i
            break
            
    return path[start_idx:], path[start_idx], path[-1]

# ==========================================
# 2. 시뮬레이션: 다양한 빌런 혈관 생성기
# ==========================================

def generate_synthetic_capillary(diameter_base=15.0):
    width, height = 250, 400
    img = np.zeros((height, width), dtype=np.uint8)
    
    # 랜덤 파라미터 (다양한 모양을 위해)
    diameter = diameter_base * random.uniform(0.8, 2.5) # 두께 변화
    apex_radius = random.uniform(25, 65)               # 굴곡 변화
    tortuosity = random.uniform(0, 12)                 # 꼬임 정도
    
    t = np.linspace(0, np.pi * 1.1, 300)
    x = width//2 + apex_radius * np.cos(t)
    y = 60 + 200 * (1 - np.sin(t))
    
    # 다리 부분에 무작위 꼬임 추가
    leg_mask = t > np.pi/2
    x[leg_mask] += tortuosity * np.sin(0.04 * y[leg_mask])
    
    pts = np.vstack((x, y)).astype(np.int32).T
    cv2.polylines(img, [pts], False, 255, int(diameter))
    
    # 노이즈 및 블러 추가
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    return binary, diameter, f"D:{diameter:.1f} R:{apex_radius:.1f} T:{tortuosity:.1f}"

# ==========================================
# 3. 메인 실행 및 통계 리포트
# ==========================================

test_count = 6
results = []
plt.figure(figsize=(16, 10))

for i in range(test_count):
    # A. 가짜 혈관 생성 (정답지 확보)
    binary_img, gt_d, info = generate_synthetic_capillary(diameter_base=15.0)
    
    # B. 뼈대 추출 및 거리 변환
    skeleton = cv2.ximgproc.thinning(binary_img)
    dist_map = distance_transform_edt(binary_img)
    
    # C. 알고리즘 적용
    ys, xs = np.where(skeleton)
    apex_ref = [np.min(ys), xs[np.argmin(ys)]]
    
    try:
        full_path = order_component_path(skeleton, apex_ref)
        trimmed_path, _, _ = find_limb_valid_range_by_curvature(full_path)
        
        # D. 두께 측정 (거리 변환 맵의 최대값 * 2)
        trimmed_coords = np.array(trimmed_path)
        measured_d = np.max(dist_map[trimmed_coords[:, 0], trimmed_coords[:, 1]]) * 2
        
        # E. 오차 및 정확도 계산
        accuracy = (1 - abs(gt_d - measured_d) / gt_d) * 100
        results.append(accuracy)
        
        # 시각화
        plt.subplot(2, 3, i+1)
        plt.imshow(binary_img, cmap='gray')
        plt.plot(trimmed_coords[:, 1], trimmed_coords[:, 0], 'r.', markersize=2, label='Measured Area')
        plt.title(f"Case {i+1} ({info})\nAccuracy: {accuracy:.2f}%")
        plt.axis('off')
        
    except Exception as e:
        print(f"Error in Case {i+1}: {e}")

# 최종 리포트
print(f"\n" + "="*30)
print(f"대량 검증 최종 결과")
print(f"평균 정확도: {np.mean(results):.2f}%")
print(f"최저 정확도: {np.min(results):.2f}%")
print(f"="*30)
plt.tight_layout()
plt.show()