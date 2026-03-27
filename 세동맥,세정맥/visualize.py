'''
결과 시각화 함수

'''
import matplotlib.pyplot as plt

def plot_and_print_results(search_name, leg_results, labeled_mask, apex_pt, used_branch_pt):
    print(f"\n{'='*50}")
    print(f"이미지 파일: {search_name}")
    print("측정 로직: 양 끝단(Apex & Branch) 굴곡 배제 (곡률 기반)")
    
    for idx, res in enumerate(leg_results):
        side = "세동맥(Arterial)" if idx == 0 else "세정맥(Venous)"
        print(f"- {side}")
        print(f"  · 측정 구간: ({res['start_pt'][1]}, {res['start_pt'][0]}) ~ ({res['end_pt'][1]}, {res['end_pt'][0]})")
        print(f"  · 최대 직경: {res['max_d']:.2f} px")
    print(f"{'='*50}\n")

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 4)
    plt.imshow(labeled_mask, cmap="nipy_spectral")

    colors = ["cyan", "lime"]
    for idx, res in enumerate(leg_results):
        side = "Arterial" if idx == 0 else "Venous"
        color = colors[idx % 2]

        # 제외된 구간
        px, py = [p[1] for p in res["path"]], [p[0] for p in res["path"]]
        plt.plot(px, py, color="lightgray", linewidth=2, zorder=1)

        # 실제 측정 구간
        tx, ty = [p[1] for p in res["trimmed_path"]], [p[0] for p in res["trimmed_path"]]
        plt.plot(tx, ty, color=color, linewidth=2, label=f"{side} Range", zorder=2)

        # 시작/끝점
        plt.scatter(res["start_pt"][1], res["start_pt"][0], c="yellow", s=30, edgecolors="black", label=f"{side} Start", zorder=3)
        plt.scatter(res["end_pt"][1], res["end_pt"][0], c="yellow", s=30, edgecolors="black", marker='s', label=f"{side} End", zorder=3)

        # 최대 직경 별
        plt.scatter(res["max_x"], res["max_y"], c="red", s=30, edgecolors="white", zorder=4)

    plt.scatter(apex_pt[1], apex_pt[0], c="white", s=30, edgecolors="black", label="Apex", zorder=5)
    if used_branch_pt:
        plt.scatter(used_branch_pt[1], used_branch_pt[0], c="white", s=30, edgecolors="black", label="Branch", zorder=5)

    plt.title("Double Curvature Trimming (Apex & Branch)")
    plt.legend(fontsize='small', loc="upper right")
    plt.tight_layout()
    plt.show()