import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# =========================================================
# 1. 스플라인 보조 함수
# =========================================================
def catmull_rom_spline(points, samples_per_seg=40):
    P = np.asarray(points, dtype=np.float32)
    curve = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            pt = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2*p0 - 5*p1 + 4*p2 - p3) * t2 + (-p0 + 3*p1 - 3*p2 + p3) * t3)
            curve.append(pt)
    curve.append(P[-2])
    return np.array(curve, dtype=np.float32)

def smooth_polyline(points, samples_per_seg=20):
    pts = np.asarray(points, dtype=np.float32)
    padded = np.vstack([pts[0], pts, pts[-1]])
    return catmull_rom_spline(padded, samples_per_seg=samples_per_seg)

# =========================================================
# 2. 임상 Crossing 모델 (형태 다양성 극대화 + 강제 타겟팅)
# =========================================================
def generate_clinical_crossing_mask(cx=256, top_y=120, bottom_y=480, art_thick=12, ven_thick=18):
    is_double_cross = random.choice([True, False])
    
    # 🌟 [다양성 1] 좌우 비대칭 너비 (Asymmetric Widths)
    b_hw_L = random.randint(20, 70) 
    b_hw_R = random.randint(20, 70)
    t_hw_L = random.randint(20, 55) 
    t_hw_R = random.randint(20, 55)
    
    # 🌟 [다양성 2] 교차점 및 정점의 무작위 이동 (Off-center)
    apex_shift = random.uniform(-25, 25)  # 꼭대기가 좌우로 쏠림
    cross_shift = random.uniform(-20, 20) # 교차점이 중심에서 벗어남
    
    # 🌟 [다양성 3] 전체적인 S자/C자 휘어짐 추가
    tilt = random.uniform(-30, 30)
    bend = random.uniform(-25, 25)
    
    lowest_cross_y = random.randint(bottom_y - 220, bottom_y - 120) 
    upper_cross_y = top_y + (lowest_cross_y - top_y) // 2 + random.randint(-15, 15)
    
    def transform(x, y):
        dist_from_bottom = (bottom_y - y) / (bottom_y - top_y + 1e-6)
        # 기울기 + S/C자형 곡선 휘어짐
        nx = x + tilt * dist_from_bottom + bend * np.sin(dist_from_bottom * np.pi)
        return (nx, y)

    # 비대칭 뼈대 구성
    if not is_double_cross: # 1번 꼬임
        cpts = [
            transform(cx - b_hw_L, bottom_y),            
            transform(cx - b_hw_L*0.6, lowest_cross_y + 40), 
            transform(cx + cross_shift, lowest_cross_y),             # 중심을 벗어난 교차점
            transform(cx + t_hw_R, lowest_cross_y - 60), 
            transform(cx + t_hw_R*0.6 + apex_shift, top_y + 20),      
            transform(cx + apex_shift, top_y),                       # 중심을 벗어난 꼭대기
            transform(cx - t_hw_L*0.6 + apex_shift, top_y + 20),      
            transform(cx - t_hw_L, lowest_cross_y - 60), 
            transform(cx + cross_shift, lowest_cross_y),             
            transform(cx + b_hw_R*0.6, lowest_cross_y + 40), 
            transform(cx + b_hw_R, bottom_y)             
        ]
        top_cross_y = lowest_cross_y 
    else: # 2번 꼬임
        cpts = [
            transform(cx - b_hw_L, bottom_y),            
            transform(cx - b_hw_L*0.5, lowest_cross_y + 40), 
            transform(cx + cross_shift, lowest_cross_y),             
            transform(cx + t_hw_R, (lowest_cross_y + upper_cross_y)//2), 
            transform(cx - cross_shift*0.5, upper_cross_y),          # 두 번째 교차점도 비대칭
            transform(cx - t_hw_L*0.8 + apex_shift, top_y + 20),      
            transform(cx + apex_shift, top_y),                      
            transform(cx + t_hw_R*0.8 + apex_shift, top_y + 20),      
            transform(cx - cross_shift*0.5, upper_cross_y),          
            transform(cx - t_hw_L, (lowest_cross_y + upper_cross_y)//2), 
            transform(cx + cross_shift, lowest_cross_y),             
            transform(cx + b_hw_R*0.5, lowest_cross_y + 40), 
            transform(cx + b_hw_R, bottom_y)             
        ]
        top_cross_y = upper_cross_y 

    gt_centerline = smooth_polyline(cpts, samples_per_seg=35) 
    img = np.zeros((512, 512), dtype=np.uint8)
    pts = np.round(gt_centerline).astype(np.int32)
    total_points = len(pts)

    # 타겟 안전 구역 및 중앙 지점 (이전의 완벽한 로직 유지!)
    safe_zone_top = top_y + 15
    safe_zone_bottom = top_cross_y - 15
    target_mid_y = (safe_zone_top + safe_zone_bottom) / 2

    # 불규칙 요철 노이즈
    freqs = [0.15, 0.4]; phases = [random.uniform(0, 2*np.pi) for _ in range(2)]
    amps = [random.uniform(1.0, 2.0) for _ in range(2)]

    # 🌟 팽창(Bulge)과 수축(Pinch) 강제 타겟팅 로직 (유지)
    def get_thickness(idx, y_pos):
        p = idx / float(total_points - 1)
        base_thickness = art_thick + (ven_thick - art_thick) * p
        wave = sum(amps[j] * np.sin(freqs[j] * idx + phases[j]) for j in range(2))
        
        bulge = 3.5 * np.exp(-((y_pos - target_mid_y)**2) / 300.0)
        
        pinch = 0
        dist_to_cross = top_cross_y - y_pos
        if 0 < dist_to_cross < 40:
            pinch = -3.5 * (1.0 - (dist_to_cross / 40.0)**2) 
            
        final_thick = base_thickness + wave + bulge + pinch
        return max(3, int(final_thick))

    max_art_thick = max_ven_thick = 0
    max_art_idx = max_ven_idx = 0

    for i in range(total_points - 1):
        current_y = pts[i][1]
        current_thickness = get_thickness(i, current_y)

        # 첫 번째 교차점 이전 상단 다리 라벨링 (유지)
        if safe_zone_top < current_y < safe_zone_bottom:
            if i < total_points // 2: 
                if current_thickness > max_art_thick:
                    max_art_thick, max_art_idx = current_thickness, i
            else:                     
                if current_thickness > max_ven_thick:
                    max_ven_thick, max_ven_idx = current_thickness, i
                
        cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), 255, current_thickness, lineType=cv2.LINE_AA)
     
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
    
    def draw_diameter_line(idx, thickness, color):
        if idx == 0 or idx >= total_points - 1: return
        p1, p2 = pts[idx], pts[idx+1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length == 0: return
        nx, ny = -dy / length, dx / length
        start_pt = (int(p1[0] + nx * thickness / 2), int(p1[1] + ny * thickness / 2))
        end_pt = (int(p1[0] - nx * thickness / 2), int(p1[1] - ny * thickness / 2))
        cv2.line(img_rgb, start_pt, end_pt, color, 3, cv2.LINE_AA)

    draw_diameter_line(max_art_idx, max_art_thick, (255, 0, 0))
    draw_diameter_line(max_ven_idx, max_ven_thick, (0, 0, 255))
    
    cross_type = 2 if is_double_cross else 1
    return img_rgb, gt_centerline, (max_art_thick, max_ven_thick), cross_type

# =========================================================
# 3. 30장 데이터셋 생성 및 시각화
# =========================================================
def generate_and_show_30_clinical_crossing_masks():
    print("🔄 [모양 다양화] 비대칭성과 굴곡이 적용된 다이내믹 30장 생성 중...")
    images, gt_info = [], []
    titles = []
    
    for i in range(30):
        rand_art = random.randint(12, 18)
        rand_ven = int(rand_art * random.uniform(1.2, 1.7))
        
        img, _, gt_thicks, cross_cnt = generate_clinical_crossing_mask(
            cx=random.randint(200, 312), top_y=random.randint(60, 120), bottom_y=490,
            art_thick=rand_art, ven_thick=rand_ven
        )
        images.append(img); gt_info.append(gt_thicks)
        titles.append(f"A:{gt_thicks[0]}, V:{gt_thicks[1]}")

    plt.figure(figsize=(18, 16)) 
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.imshow(images[i]) 
        plt.title(f"S{i+1}: {titles[i]}", fontsize=11, fontweight='bold')
        plt.axis("off")
    plt.tight_layout(h_pad=3.0, w_pad=1.0); plt.show()
    print("✅ 타점의 안정성과 형태의 다양성을 모두 잡은 데이터셋 완성!")

if __name__ == "__main__":
    SEED_VALUE = 42; random.seed(SEED_VALUE); np.random.seed(SEED_VALUE)
    generate_and_show_30_clinical_crossing_masks()