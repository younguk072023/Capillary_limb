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
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            curve.append(pt)
    curve.append(P[-2])
    return np.array(curve, dtype=np.float32)


def smooth_polyline(points, samples_per_seg=25):
    pts = np.asarray(points, dtype=np.float32)
    padded = np.vstack([pts[0], pts, pts[-1]])
    return catmull_rom_spline(padded, samples_per_seg=samples_per_seg)


# =========================================================
# 2. 길이 계산
# =========================================================
def calc_polyline_length(curve):
    if len(curve) < 2:
        return 0.0
    d = np.diff(curve, axis=0)
    seg = np.sqrt(np.sum(d * d, axis=1))
    return float(np.sum(seg))


# =========================================================
# 3. Tortuous wave 함수
#    - 단일 sine이 아니라 복합파 + 구간별 damping
# =========================================================
def tortuous_offset(t, amp_main, amp_sub, phase_main, phase_sub, n_main, n_sub):
    """
    t: 0(bottom) ~ 1(top)
    아래는 거의 직선, 위로 갈수록 구불거림 증가
    """
    # 하단은 straight, 중간부터 증가, 상단에서 가장 강해짐
    if t < 0.15:
        damping = 0.0
    elif t < 0.35:
        damping = (t - 0.15) / 0.20 * 0.45
    else:
        damping = 0.45 + (t - 0.35) / 0.65 * 0.55

    # 큰 굴곡 + 작은 굴곡 섞기
    main_wave = amp_main * np.sin(n_main * np.pi * t + phase_main)
    sub_wave = amp_sub * np.sin(n_sub * np.pi * t + phase_sub)

    # 너무 규칙적인 수학곡선 티 줄이기
    asym_bias = 4.0 * np.sin(0.8 * np.pi * t + 0.3)

    return (main_wave + sub_wave + asym_bias) * damping


# =========================================================
# 4. 임상 Tortuous 마스크 생성
#    - 예시 그림처럼 crossing 없이 구불구불한 형태
#    - 라벨은 하단 독립 limb에서만 찾음
# =========================================================
def generate_clinical_tortuous_mask(
    cx=256,
    top_y=85,
    bottom_y=490,
    art_thick=18,
    ven_thick=28,
    canvas_size=512
):
    # 아래 폭 / 위 폭
    b_hw_L = random.randint(32, 48)
    b_hw_R = random.randint(32, 48)
    t_hw_L = random.randint(18, 30)
    t_hw_R = random.randint(18, 30)

    # 큰 굴곡 수를 줄여서 "잔물결"보다 "구불거림" 느낌 강조
    n_main = random.uniform(2.0, 3.4)
    n_sub = random.uniform(4.8, 6.5)

    # 좌우 다리 amplitude
    amp_main_L = random.uniform(18, 30)
    amp_main_R = random.uniform(18, 32)
    amp_sub_L = random.uniform(5, 11)
    amp_sub_R = random.uniform(5, 12)

    # 좌우 위상은 비슷하지만 완전히 같지 않게
    phase_main_L = random.uniform(0, 2 * np.pi)
    phase_main_R = phase_main_L + random.uniform(-0.45, 0.45)
    phase_sub_L = random.uniform(0, 2 * np.pi)
    phase_sub_R = phase_sub_L + random.uniform(-0.55, 0.55)

    cpts = []

    # -----------------------------------------------------
    # 왼쪽 limb (arterial side 가정)
    # -----------------------------------------------------
    left_pts = []
    n_left = 9
    for i in range(n_left):
        t = i / (n_left - 1)   # 0(bottom) ~ 1(top)
        y = bottom_y - t * (bottom_y - top_y)

        base_x = cx - (b_hw_L * (1 - t) + t_hw_L * t)

        wave_x = tortuous_offset(
            t=t,
            amp_main=amp_main_L,
            amp_sub=amp_sub_L,
            phase_main=phase_main_L,
            phase_sub=phase_sub_L,
            n_main=n_main,
            n_sub=n_sub
        )

        # 위로 갈수록 약간 안쪽으로 말리게
        inward_pull = 5.0 * (t ** 1.6)

        x = base_x + wave_x + inward_pull
        left_pts.append((x, y))

    # apex 전 마지막 점 보정
    left_pts[-1] = (cx - random.uniform(10, 16), top_y + random.uniform(10, 18))

    # -----------------------------------------------------
    # apex
    #   너무 완벽한 반원 말고 약간 찌그러진 꼭대기
    # -----------------------------------------------------
    apex_mid = (cx + random.uniform(-4, 4), top_y + random.uniform(-2, 2))
    apex_right = (cx + random.uniform(10, 16), top_y + random.uniform(10, 18))

    # -----------------------------------------------------
    # 오른쪽 limb (venous side 가정)
    # -----------------------------------------------------
    right_pts = []
    n_right = 8
    for i in range(n_right - 1, -1, -1):
        t = i / (n_right - 1)   # 1(top) -> 0(bottom) 용으로 뒤집어서 append
        y = bottom_y - t * (bottom_y - top_y)

        base_x = cx + (b_hw_R * (1 - t) + t_hw_R * t)

        wave_x = tortuous_offset(
            t=t,
            amp_main=amp_main_R,
            amp_sub=amp_sub_R,
            phase_main=phase_main_R,
            phase_sub=phase_sub_R,
            n_main=n_main,
            n_sub=n_sub
        )

        inward_pull = 5.0 * (t ** 1.6)

        x = base_x + wave_x - inward_pull
        right_pts.append((x, y))

    right_pts[0] = apex_right

    # -----------------------------------------------------
    # crossing 방지용 최소 간격 보정
    # -----------------------------------------------------
    # 동일 y 레벨에서 좌우가 너무 가까워지면 오른쪽을 조금 밀어냄
    min_gap = random.uniform(26, 34)
    for k in range(min(len(left_pts), len(right_pts))):
        lx, ly = left_pts[k]
        rx, ry = right_pts[-1 - k]

        # y가 비슷한 점들끼리 비교
        if abs(ly - ry) < 18:
            if rx - lx < min_gap:
                shift = (min_gap - (rx - lx)) / 2.0 + 1.0
                left_pts[k] = (lx - shift, ly)
                right_pts[-1 - k] = (rx + shift, ry)

    # control point 결합
    cpts.extend(left_pts)
    cpts.append(apex_mid)
    cpts.extend(right_pts)

    # spline 중심선
    gt_centerline = smooth_polyline(cpts, samples_per_seg=32)
    pts = np.round(gt_centerline).astype(np.int32)

    # -----------------------------------------------------
    # 두께 프로파일
    #   - 아래 safe zone에서 가장 두껍게
    #   - 위 tortuous zone은 상대적으로 살짝 얇게
    # -----------------------------------------------------
    total_points = len(pts)

    safe_zone_top = bottom_y - 130
    safe_zone_bottom = bottom_y - 20
    target_mid_y = (safe_zone_top + safe_zone_bottom) / 2.0

    freqs = [0.11, 0.31]
    phases = [random.uniform(0, 2*np.pi) for _ in range(2)]
    amps = [random.uniform(0.8, 1.8), random.uniform(0.8, 1.6)]

    def get_thickness(idx, y_pos):
        p = idx / float(max(total_points - 1, 1))

        # 왼쪽은 좀 얇고 오른쪽은 좀 두껍게
        base = art_thick + (ven_thick - art_thick) * p

        wave = sum(amps[j] * np.sin(freqs[j] * idx + phases[j]) for j in range(2))

        # safe zone 중앙 bulge
        bulge = 5.5 * np.exp(-((y_pos - target_mid_y) ** 2) / 420.0)

        # 위쪽 tortuous zone은 약간 얇게
        pinch = 0.0
        if y_pos < safe_zone_top:
            pinch = -3.5
        elif safe_zone_top <= y_pos <= safe_zone_top + 35:
            pinch = -3.5 * (1 - (y_pos - safe_zone_top) / 35.0)

        final_thick = base + wave + bulge + pinch
        return max(5, int(round(final_thick)))

    # -----------------------------------------------------
    # 렌더링
    # -----------------------------------------------------
    img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    max_art_thick = 0
    max_ven_thick = 0
    max_art_idx = 0
    max_ven_idx = total_points - 1

    for i in range(total_points - 1):
        y = pts[i][1]
        thickness = get_thickness(i, y)

        # 하단 safe zone 안에서만 라벨 탐색
        if safe_zone_top < y < safe_zone_bottom:
            if i < total_points // 2:
                if thickness > max_art_thick:
                    max_art_thick = thickness
                    max_art_idx = i
            else:
                if thickness > max_ven_thick:
                    max_ven_thick = thickness
                    max_ven_idx = i

        cv2.line(
            img,
            tuple(pts[i]),
            tuple(pts[i + 1]),
            255,
            thickness,
            lineType=cv2.LINE_AA
        )

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)

    # -----------------------------------------------------
    # 직경선 표시
    # -----------------------------------------------------
    def draw_diameter_line_normal(idx, thickness, color):
        if idx <= 0 or idx >= total_points - 1:
            return
        p1 = pts[idx - 1].astype(np.float32)
        p2 = pts[idx + 1].astype(np.float32)

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length == 0:
            return

        nx, ny = -dy / length, dx / length
        c = pts[idx].astype(np.float32)

        start_pt = (int(round(c[0] + nx * thickness / 2)), int(round(c[1] + ny * thickness / 2)))
        end_pt   = (int(round(c[0] - nx * thickness / 2)), int(round(c[1] - ny * thickness / 2)))

        cv2.line(img_rgb, start_pt, end_pt, color, 3, cv2.LINE_AA)

    draw_diameter_line_normal(max_art_idx, max_art_thick, (255, 0, 0))  # arterial
    draw_diameter_line_normal(max_ven_idx, max_ven_thick, (0, 0, 255))  # venous

    gt_info = {
        "art_max_thick": max_art_thick,
        "ven_max_thick": max_ven_thick,
        "art_max_idx": max_art_idx,
        "ven_max_idx": max_ven_idx,
        "centerline_length": calc_polyline_length(gt_centerline),
    }

    return img_rgb, final_mask, gt_centerline, gt_info


# =========================================================
# 5. 여러 장 시각화
# =========================================================
def generate_and_show_30_clinical_tortuous_masks():
    print("🔄 예시 그림 스타일에 가까운 tortuous capillary 30장 생성 중...")

    images = []
    infos = []

    for i in range(30):
        rand_art = random.randint(18, 24)
        rand_ven = int(rand_art * random.uniform(1.25, 1.65))

        img_rgb, _, _, gt_info = generate_clinical_tortuous_mask(
            cx=random.randint(205, 307),
            top_y=random.randint(70, 115),
            bottom_y=490,
            art_thick=rand_art,
            ven_thick=rand_ven
        )

        images.append(img_rgb)
        infos.append(gt_info)

    plt.figure(figsize=(18, 16))
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.imshow(images[i])
        plt.title(
            f"S{i+1}: A:{infos[i]['art_max_thick']}, V:{infos[i]['ven_max_thick']}",
            fontsize=10,
            fontweight="bold"
        )
        plt.axis("off")

    plt.tight_layout(h_pad=3.0, w_pad=1.0)
    plt.show()\

    print("✅ 하단은 비교적 독립 limb, 상단은 불규칙 tortuous 형태로 생성 완료")


# =========================================================
# 실행
# =========================================================
if __name__ == "__main__":
    SEED_VALUE = 42
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)

    generate_and_show_30_clinical_tortuous_masks()