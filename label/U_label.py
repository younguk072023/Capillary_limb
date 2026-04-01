import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# (스플라인 함수 동일)
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
# 업그레이드: 예쁜 U자(Normal) + 다양한 변이(Diverse) 혼합 생성
# =========================================================
def generate_realistic_normal_u(cx=256, top_y=120, bottom_y=480, art_thick=12, ven_thick=18, is_perfect=False):
    # 🌟 [Perfect Normal 모드] 예쁜 U자를 위해 변수 통제
    if is_perfect:
        bottom_half_width = top_half_width = random.randint(35, 45)
        tilt = 0; bend_power = 0; apex_height = 25; jitter_range = 0
    else:
        bottom_half_width = random.randint(30, 90)
        top_half_width = random.randint(15, 50)
        tilt = random.uniform(-60, 60)
        bend_power = random.uniform(-40, 40)
        apex_height = random.randint(10, 40)
        jitter_range = 8

    mid_y1 = bottom_y - int((bottom_y - top_y) * 0.3)
    mid_y2 = bottom_y - int((bottom_y - top_y) * 0.7)

    def transform_pt(x, y, apply_jitter=True):
        dist_from_bottom = (bottom_y - y) / (bottom_y - top_y + 1e-6)
        new_x = x + (tilt * dist_from_bottom) + (bend_power * np.sin(dist_from_bottom * np.pi))
        if apply_jitter and not is_perfect:
            new_x += random.uniform(-jitter_range, jitter_range)
            y += random.uniform(-jitter_range, jitter_range)
        return (new_x, y)

    control_points = [
        transform_pt(cx - bottom_half_width, bottom_y), 
        transform_pt(cx - (bottom_half_width + top_half_width)/2 - 5, mid_y1), 
        transform_pt(cx - top_half_width - 8, mid_y2),     
        transform_pt(cx - top_half_width//2, top_y + apex_height), 
        transform_pt(cx, top_y, apply_jitter=False), 
        transform_pt(cx + top_half_width//2, top_y + apex_height),               
        transform_pt(cx + top_half_width + 8, mid_y2), 
        transform_pt(cx + (bottom_half_width + top_half_width)/2 + 5, mid_y1), 
        transform_pt(cx + bottom_half_width, bottom_y)         
    ]
    
    gt_centerline = smooth_polyline(control_points, samples_per_seg=25)
    img = np.zeros((512, 512), dtype=np.uint8)
    pts = np.round(gt_centerline).astype(np.int32)
    total_points = len(pts)
    
    # [복합 요철] Perfect 모드에서는 요철 강도를 낮춤
    base_avoid = lambda p: 1.0 - 0.28 * np.sin(p * np.pi) 
    amp_scale = 0.5 if is_perfect else 1.0
    freqs = [0.12, 0.35, 0.6]; phases = [random.uniform(0, 2*np.pi) for _ in range(3)]
    amps = [random.uniform(1.0, 3.0) * amp_scale for _ in range(3)]

    def get_complex_fluctuation(idx):
        wave_sum = sum(amps[j] * np.sin(freqs[j] * idx + phases[j]) for j in range(3))
        return base_avoid(idx / (total_points - 1)), wave_sum

    max_art_thick = max_art_idx = max_ven_thick = max_ven_idx = 0

    for i in range(total_points - 1):
        base_factor, complex_wave = get_complex_fluctuation(i)
        base_thickness = (art_thick + (ven_thick - art_thick) * (i/(total_points-1))) * base_factor
        current_thickness = int(base_thickness + complex_wave)
        current_thickness = max(4, current_thickness) 

        if i < total_points // 2:
            if current_thickness > max_art_thick: max_art_thick, max_art_idx = current_thickness, i
        else:
            if current_thickness > max_ven_thick: max_ven_thick, max_ven_idx = current_thickness, i
                
        cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), 255, current_thickness, lineType=cv2.LINE_AA)
    
    blurred = cv2.GaussianBlur(img, (7, 7), 0); _, final_img = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    
    def draw_diameter_line(idx, thickness, color):
        p1, p2 = pts[idx], pts[idx+1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(dx, dy); nx, ny = -dy / (length+1e-6), dx / (length+1e-6)
        start_pt = (int(p1[0] + nx * thickness / 2), int(p1[1] + ny * thickness / 2))
        end_pt = (int(p1[0] - nx * thickness / 2), int(p1[1] - ny * thickness / 2))
        cv2.line(img_rgb, start_pt, end_pt, color, 3, cv2.LINE_AA)

    draw_diameter_line(max_art_idx, max_art_thick, (255, 0, 0))
    draw_diameter_line(max_ven_idx, max_ven_thick, (0, 0, 255))
    
    return img_rgb, gt_centerline, (max_art_thick, max_ven_thick)

def generate_and_show_30_realistic_samples():
    print("🔄 예쁜 정상형이 포함된 초현실적 30장을 생성 중입니다...")
    images, gt_info = [], []
    for i in range(30):
        # 🌟 15% 확률로 아주 예쁜 'Perfect Normal' 생성
        is_perfect = True if random.random() < 0.15 else False
        rand_art = random.randint(12, 24); rand_ven = int(rand_art * random.uniform(1.2, 1.7))
        
        img, _, gt_thicks = generate_realistic_normal_u(
            cx=random.randint(200, 312), top_y=random.randint(60, 180), 
            art_thick=rand_art, ven_thick=rand_ven, is_perfect=is_perfect
        )
        images.append(img); gt_info.append(gt_thicks)

    plt.figure(figsize=(18, 16)) 
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.imshow(images[i]) 
        plt.title(f"S{i+1}: A:{gt_info[i][0]}, V:{gt_info[i][1]}", fontsize=11, fontweight='bold')
        plt.axis("off")
    plt.tight_layout(h_pad=3.0, w_pad=1.0); plt.show()

if __name__ == "__main__":
    SEED_VALUE = 42; random.seed(SEED_VALUE); np.random.seed(SEED_VALUE)
    generate_and_show_30_realistic_samples()