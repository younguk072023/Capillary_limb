'''
수동 라벨 알고리즘 작성
전문가 사용자가 이미지에서 혈관의 직경을 측정하기 위해 4개의 점을 찍는 과정
'''
import os
import cv2
import csv
import math
import numpy as np

IMAGE_DIR = r"C:\Users\park_younguk\Desktop\limb_label"  
CSV_FILENAME = os.path.join(IMAGE_DIR, "manual_measurement_results.csv")

current_points = []
img_display = None
img_clean = None
results_dict = {} 
image_files = []
current_img_index = 0

def read_image_unicode(image_path):
    img_array = np.fromfile(image_path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def calculate_distance(p1, p2):
    #두 점 사이의 유클리드 거리 공식으로 계산
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def save_to_csv():
    """알고리즘 출력 파일과 Column Name을 완벽히 일치시켜 저장"""
    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", 
            "A_x1", "A_y1", "A_x2", "A_y2", "Manual_Arterial_Diameter(px)", 
            "V_x1", "V_y1", "V_x2", "V_y2", "Manual_Venous_Diameter(px)"
        ])
        
        for img_name, pts in results_dict.items():
            if len(pts) == 4:
                d_a = calculate_distance(pts[0], pts[1]) # 첫 번째 쌍 (Arterial)
                d_v = calculate_distance(pts[2], pts[3]) # 두 번째 쌍 (Venous)
                row = [
                    img_name, 
                    pts[0][0], pts[0][1], pts[1][0], pts[1][1], round(d_a, 2),
                    pts[2][0], pts[2][1], pts[3][0], pts[3][1], round(d_v, 2)
                ]
                writer.writerow(row)

def draw_overlay():
    """화면에 측정 가이드 라인과 텍스트 표시"""
    global img_display
    img_display = img_clean.copy()
    
    total_imgs = len(image_files)
    current_name = image_files[current_img_index]
    status_text = f"[{current_img_index + 1}/{total_imgs}] {current_name}"
    cv2.putText(img_display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Arterial  빨간색
    if len(current_points) >= 1: cv2.circle(img_display, current_points[0], 2, (0, 0, 255), -1)
    if len(current_points) >= 2:
        cv2.circle(img_display, current_points[1], 2, (0, 0, 255), -1)
        cv2.line(img_display, current_points[0], current_points[1], (0, 0, 255), 2)
        dist_a = calculate_distance(current_points[0], current_points[1])
        cv2.putText(img_display, f"A (Left): {dist_a:.2f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Venous 파란색
    if len(current_points) >= 3: cv2.circle(img_display, current_points[2], 2, (255, 0, 0), -1)
    if len(current_points) >= 4:
        cv2.circle(img_display, current_points[3], 2, (255, 0, 0), -1)
        cv2.line(img_display, current_points[2], current_points[3], (255, 0, 0), 2)
        dist_v = calculate_distance(current_points[2], current_points[3])
        cv2.putText(img_display, f"V (Right): {dist_v:.2f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    if len(current_points) == 4:
        cv2.putText(img_display, "Press ENTER to save", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capillary Measurement Tool", img_display)

def mouse_callback(event, x, y, flags, param):
    global current_points
    # 좌클릭: 점 추가
    if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
        current_points.append((x, y))
        draw_overlay()
    # 우클릭: 마지막 점 지우기
    elif event == cv2.EVENT_RBUTTONDOWN and len(current_points) > 0:
        current_points.pop()
        draw_overlay()

def load_existing_csv():
    """이미 작업된 CSV가 있으면 데이터를 results_dict로 로드"""
    if os.path.exists(CSV_FILENAME):
        try:
            with open(CSV_FILENAME, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader, None) # 헤더 스킵
                for row in reader:
                    if len(row) >= 11:
                        img_name = row[0]
                        pts = [(int(float(row[1])), int(float(row[2]))), (int(float(row[3])), int(float(row[4]))),
                               (int(float(row[6])), int(float(row[7]))), (int(float(row[8])), int(float(row[9])))]
                        results_dict[img_name] = pts
            print(f"기존 데이터 {len(results_dict)}건 로드 완료.")
        except Exception as e:
            print(f"로드 중 오류 발생: {e}")

def main():
    global img_display, img_clean, current_points, image_files, current_img_index
    
    if not os.path.exists(IMAGE_DIR):
        print(f"폴더 없음: {IMAGE_DIR}")
        return
        
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.png')]
    if not image_files:
        print("이미지가 없습니다.")
        return
        
    load_existing_csv()
    
    cv2.namedWindow("Capillary Measurement Tool")
    cv2.setMouseCallback("Capillary Measurement Tool", mouse_callback)

    print("\n[단축키]")
    print(" - 마우스 좌클릭 : 점 찍기")
    print(" - 마우스 우클릭 / 'z' : 마지막 점 지우기(Undo)")
    print(" - 엔터 / 스페이스 : 저장 및 다음 이미지")
    print(" - 'b' : 이전 이미지로 이동")
    print(" - 'r' : 현재 이미지 모든 점 초기화")
    print(" - ESC / 'q' : 종료")

    while current_img_index < len(image_files):
        img_name = image_files[current_img_index]
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_clean = read_image_unicode(img_path)
        
        # 이미 데이터가 있으면 가져오고, 없으면 빈 리스트
        current_points = results_dict.get(img_name, []).copy()
        draw_overlay()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # 다음으로 (Enter/Space)
            if key == 13 or key == 32:
                if len(current_points) == 4:
                    results_dict[img_name] = current_points.copy()
                    save_to_csv()
                    current_img_index += 1
                    break
                else:
                    print("4개를 찍어야 저장 가능합니다.")
            
            # 이전으로 (b)
            elif key == ord('b') and current_img_index > 0:
                results_dict[img_name] = current_points.copy() # 현재 것도 임시 저장
                current_img_index -= 1
                break
                
            # 마지막 점 취소 (z)
            elif key == ord('z'):
                if len(current_points) > 0:
                    current_points.pop()
                    draw_overlay()
                    
            # 리셋 (r)
            elif key == ord('r'):
                current_points = []
                draw_overlay()
                
            # 종료 (ESC/q)
            elif key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"작업 완료! 결과: {CSV_FILENAME}")

if __name__ == "__main__":
    main()