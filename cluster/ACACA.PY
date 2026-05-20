"""
논문용 단일 이미지 기하학 알고리즘 시각화 코드

목적:
- 네 기존 기하학 알고리즘 analyze_single_image()를 그대로 적용
- p23_det_006.png에 대해 U/D keypoint 표시
- 알고리즘이 산출한 세동맥/세정맥 최대 직경선을 표시
- 가능하면 leg_results 내부의 skeleton/centerline 좌표도 빨강/파랑으로 표시

색상:
- 혈관 mask: 흰색
- U point: 초록
- D point: 노랑
- 세동맥측: 빨강
- 세정맥측: 파랑
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. 프로젝트 경로 설정
# =========================================================
PROJECT_ROOT = r"C:\Users\park_younguk\Desktop\Capillary_limb"
LIMB_DIR = os.path.join(PROJECT_ROOT, "limb")

# analyzer.py가 limb 폴더 안에 있으므로 추가
if LIMB_DIR not in sys.path:
    sys.path.insert(0, LIMB_DIR)

# 혹시 analyzer.py 내부에서 다른 파일을 import할 때를 대비해 project root도 추가
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from analyzer import analyze_single_image
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "\n[오류] analyzer.py를 import하지 못했습니다.\n"
        f"현재 설정된 LIMB_DIR: {LIMB_DIR}\n"
        "analyzer.py가 실제로 C:\\Users\\park_younguk\\Desktop\\Capillary_limb\\limb 안에 있는지 확인하세요.\n"
        "지금 구조상 거기가 맞아 보이긴 함. 파이썬만 못 알아먹는 중.\n"
    ) from e


# =========================================================
# 1. 경로 설정
# =========================================================
IMAGE_PATH = r"C:\Users\park_younguk\Desktop\Capillary_limb\cluster\p23_det_006.png"
KEYPOINT_CSV = r"C:\Users\park_younguk\Desktop\effi_corr\efficientNet-b2.csv"

SAVE_PATH = r"C:\Users\park_younguk\Desktop\Capillary_limb\cluster\p23_det_006_algorithm_visual.png"


# =========================================================
# 2. 시각화 설정
# =========================================================
# RGB 기준 색상
U_COLOR = (0, 255, 0)          # 초록
D_COLOR = (255, 255, 0)        # 노랑
ART_COLOR = (255, 0, 0)        # 세동맥측: 빨강
VEN_COLOR = (0, 80, 255)       # 세정맥측: 파랑

POINT_RADIUS = 6
DIAMETER_LINE_THICKNESS = 2
SKELETON_RADIUS = 1
DPI = 600

# 네 기존 batch 코드와 동일한 방식
# sorted_legs_by_size[0] = arterial
# sorted_legs_by_size[1] = venous
# leg가 3개 이상일 때 가장 큰 leg를 venous로 쓰려면 False로 바꿔도 됨
EXACT_BATCH_LOGIC = True


# =========================================================
# 3. 유틸 함수
# =========================================================
def read_image_unicode(path, flags=cv2.IMREAD_COLOR):
    """한글 경로 대응 이미지 읽기"""
    img_array = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_array, flags)
    return img


def save_rgb_figure(path, rgb_img, dpi=600):
    """RGB 이미지를 고해상도 저장"""
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_keypoint_dataframe(csv_path):
    """
    keypoint CSV 로드
    - pred_U_x 형태와 U_x 형태 둘 다 대응
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    df = df_raw.rename(columns={
        "pred_U_x": "U_x",
        "pred_U_y": "U_y",
        "pred_D_x": "D_x",
        "pred_D_y": "D_y",
        "filename": "filename"
    }).copy()

    if "filename" not in df.columns:
        raise KeyError("CSV에 filename 컬럼이 없습니다.")

    required_cols = ["U_x", "U_y", "D_x", "D_y"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise KeyError(f"CSV에 필요한 좌표 컬럼이 없습니다: {missing}")

    df["filename"] = df["filename"].astype(str).str.strip()
    df["filename_only"] = df["filename"].apply(lambda x: os.path.basename(str(x)))
    df["pure_filename"] = df["filename_only"].apply(
        lambda x: os.path.splitext(str(x))[0]
    )

    return df


def get_ud_points(df_keypoints, image_path):
    """이미지 파일명과 CSV를 매칭해 U, D 좌표 반환"""
    target_filename = os.path.basename(image_path)
    target_pure = os.path.splitext(target_filename)[0]

    matched = df_keypoints[df_keypoints["filename_only"] == target_filename]

    if matched.empty:
        matched = df_keypoints[df_keypoints["pure_filename"] == target_pure]

    if matched.empty:
        raise ValueError(
            f"CSV에서 해당 이미지의 U/D 좌표를 찾지 못했습니다: {target_filename}"
        )

    row = matched.iloc[0]

    U = (int(round(row["U_x"])), int(round(row["U_y"])))
    D = (int(round(row["D_x"])), int(round(row["D_y"])))

    return U, D


def make_white_vessel_canvas(image_path):
    """
    논문용 기본 canvas 생성
    - 검정 배경
    - 흰색 혈관
    """
    gray = read_image_unicode(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ValueError(f"이미지를 불러오지 못했습니다: {image_path}")

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[binary > 0] = (255, 255, 255)

    return canvas, binary


def get_diameter_endpoints_from_leg(leg):
    """
    기존 batch 코드와 동일한 방식으로 최대 직경선 양 끝점 계산
    """
    required_keys = [
        "max_x",
        "max_y",
        "local_perp_v",
        "max_d_pos",
        "max_d_neg",
        "max_d"
    ]

    missing = [k for k in required_keys if k not in leg]

    if missing:
        raise KeyError(f"leg_result에 필요한 key가 없습니다: {missing}")

    cx = float(leg["max_x"])
    cy = float(leg["max_y"])

    perp = leg["local_perp_v"]
    px = float(perp[0])
    py = float(perp[1])

    d_pos = float(leg["max_d_pos"])
    d_neg = float(leg["max_d_neg"])

    x1 = int(round(cx + px * d_pos))
    y1 = int(round(cy + py * d_pos))

    x2 = int(round(cx - px * d_neg))
    y2 = int(round(cy - py * d_neg))

    return (x1, y1), (x2, y2), float(leg["max_d"])


def extract_points_if_exist(leg):
    """
    analyze_single_image()가 leg별 skeleton/centerline 좌표를 반환하는 경우만 사용.

    analyzer.py 내부 반환 key명을 모르는 상태이므로 가능한 후보 key들을 탐색함.
    없으면 None 반환.
    """
    candidate_keys = [
        "skel_points",
        "skeleton_points",
        "centerline_points",
        "centerline_pts",
        "branch_points",
        "leg_points",
        "path_points",
        "ordered_points",
        "coords",
        "points"
    ]

    for key in candidate_keys:
        if key not in leg:
            continue

        data = leg[key]

        # numpy array
        if isinstance(data, np.ndarray):
            arr = data

            # 좌표 배열: N x 2
            if arr.ndim == 2 and arr.shape[1] == 2:
                return [(int(round(x)), int(round(y))) for x, y in arr]

            # mask 형태: H x W
            if arr.ndim == 2:
                yy, xx = np.where(arr > 0)
                return list(zip(xx, yy))

        # list / tuple
        if isinstance(data, (list, tuple)) and len(data) > 0:
            points = []

            try:
                for p in data:
                    if len(p) >= 2:
                        points.append((int(round(p[0])), int(round(p[1]))))

                if len(points) > 0:
                    return points

            except Exception:
                pass

    return None


def draw_points(canvas, points, color, radius=1):
    """좌표 리스트를 canvas에 점으로 표시"""
    if points is None:
        return

    h, w, _ = canvas.shape

    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(
                canvas,
                (x, y),
                radius,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA
            )


def draw_algorithm_result(canvas, U, D, arterial_leg, venous_leg):
    """
    네 알고리즘 결과를 그대로 시각화
    """
    # -----------------------------------------------------
    # 1. leg별 skeleton 좌표가 반환된 경우 skeleton 표시
    # -----------------------------------------------------
    arterial_points = extract_points_if_exist(arterial_leg)
    venous_points = extract_points_if_exist(venous_leg)

    if arterial_points is not None:
        draw_points(canvas, arterial_points, ART_COLOR, radius=SKELETON_RADIUS)

    if venous_points is not None:
        draw_points(canvas, venous_points, VEN_COLOR, radius=SKELETON_RADIUS)

    if arterial_points is None or venous_points is None:
        print("\n[알림]")
        print("leg_results 안에서 skeleton/centerline 좌표를 찾지 못했습니다.")
        print("따라서 현재 이미지는 네 알고리즘의 최대 직경선만 정확히 표시합니다.")
        print("빨강/파랑 skeleton까지 정확히 표시하려면 analyzer.py에서 leg별 skeleton 좌표를 반환해야 합니다.")
        print("\n현재 arterial_leg keys:")
        print(list(arterial_leg.keys()))
        print("\n현재 venous_leg keys:")
        print(list(venous_leg.keys()))
        print()

    # -----------------------------------------------------
    # 2. 최대 직경선 표시
    # -----------------------------------------------------
    a1, a2, a_d = get_diameter_endpoints_from_leg(arterial_leg)
    v1, v2, v_d = get_diameter_endpoints_from_leg(venous_leg)

    cv2.line(
        canvas,
        a1,
        a2,
        ART_COLOR,
        DIAMETER_LINE_THICKNESS,
        lineType=cv2.LINE_AA
    )

    cv2.line(
        canvas,
        v1,
        v2,
        VEN_COLOR,
        DIAMETER_LINE_THICKNESS,
        lineType=cv2.LINE_AA
    )

    # -----------------------------------------------------
    # 3. U, D 점 표시
    # -----------------------------------------------------
    cv2.circle(
        canvas,
        U,
        POINT_RADIUS,
        U_COLOR,
        thickness=-1,
        lineType=cv2.LINE_AA
    )

    cv2.circle(
        canvas,
        D,
        POINT_RADIUS,
        D_COLOR,
        thickness=-1,
        lineType=cv2.LINE_AA
    )

    return a_d, v_d


# =========================================================
# 4. main
# =========================================================
def main():
    print("네 기하학 알고리즘 기반 논문용 단일 이미지 시각화 시작")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {IMAGE_PATH}")

    # -----------------------------------------------------
    # 1. CSV 로드
    # -----------------------------------------------------
    df_keypoints = load_keypoint_dataframe(KEYPOINT_CSV)

    # -----------------------------------------------------
    # 2. U/D 좌표 읽기
    # -----------------------------------------------------
    U, D = get_ud_points(df_keypoints, IMAGE_PATH)

    print(f"U point: {U}")
    print(f"D point: {D}")

    # -----------------------------------------------------
    # 3. 네 실제 기하학 알고리즘 적용
    # -----------------------------------------------------
    result = analyze_single_image(IMAGE_PATH, df_keypoints)

    print("\nresult keys:")
    print(list(result.keys()))

    if not result.get("ok", False):
        raise RuntimeError(f"analyze_single_image 실패: {result}")

    leg_results = result.get("leg_results", [])

    if len(leg_results) < 2:
        raise RuntimeError(
            f"검출된 leg 수가 2개 미만입니다. leg_results={len(leg_results)}"
        )

    print(f"\n검출된 leg 수: {len(leg_results)}")

    # -----------------------------------------------------
    # 4. 기존 batch 코드와 동일하게 max_d 기준 정렬
    # -----------------------------------------------------
    sorted_legs_by_size = sorted(leg_results, key=lambda x: x["max_d"])

    arterial_leg = sorted_legs_by_size[0]

    if EXACT_BATCH_LOGIC:
        venous_leg = sorted_legs_by_size[1]
    else:
        venous_leg = sorted_legs_by_size[-1]

    print(f"Arterial diameter: {arterial_leg['max_d']:.2f} px")
    print(f"Venous diameter  : {venous_leg['max_d']:.2f} px")

    # -----------------------------------------------------
    # 5. 흰색 혈관 canvas 생성
    # -----------------------------------------------------
    canvas, binary = make_white_vessel_canvas(IMAGE_PATH)

    # -----------------------------------------------------
    # 6. 알고리즘 결과 시각화
    # -----------------------------------------------------
    a_d, v_d = draw_algorithm_result(
        canvas=canvas,
        U=U,
        D=D,
        arterial_leg=arterial_leg,
        venous_leg=venous_leg
    )

    # -----------------------------------------------------
    # 7. 저장
    # -----------------------------------------------------
    save_rgb_figure(SAVE_PATH, canvas, dpi=DPI)

    # -----------------------------------------------------
    # 8. 화면 표시
    # -----------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    print("\n저장 완료:")
    print(SAVE_PATH)


if __name__ == "__main__":
    main()