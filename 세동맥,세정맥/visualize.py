'''
결과 시각화 함수

'''
import matplotlib.pyplot as plt

def plot_and_print_results(search_name, leg_results, labeled_mask, apex_pt, used_branch_pt):
    print(f"\n{'='*50}")
    print(f"이미지 파일: {search_name}")

    sorted_leg_results = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)

    for idx, res in enumerate(sorted_leg_results):
        side = "세정맥(Venous)" if idx == 0 else "세동맥(Arterial)"
        print(f"- {side}")
        print(f"  · 측정 구간: ({res['start_pt'][1]}, {res['start_pt'][0]}) ~ ({res['end_pt'][1]}, {res['end_pt'][0]})")
        print(f"  · 최대 직경: {res['max_d']:.2f} px")
    print(f"{'='*50}\n")

    plt.figure(figsize=(8, 8))
    plt.imshow(labeled_mask, cmap="nipy_spectral")

    colors = ["#4C72B0", "#8C6D31"]

    for idx, res in enumerate(leg_results):
        side = "Arterial" if idx == 0 else "Venous"
        color = colors[idx % 2]

        # 전체 path 
        px = [p[1] for p in res["path"]]
        py = [p[0] for p in res["path"]]
        plt.plot(px, py, color="white", linewidth=2, zorder=1)

        # 실제 측정 구간
        tx = [p[1] for p in res["trimmed_path"]]
        ty = [p[0] for p in res["trimmed_path"]]
        plt.plot(tx, ty, color="black", linewidth=3, label=f"{side} Range", zorder=2)

        # 시작점 / 끝점
        plt.scatter(
            res["start_pt"][1], res["start_pt"][0],
            c="yellow", s=30, edgecolors="black",
            label=f"{side} Start" if idx == 0 else None,
            zorder=3
        )
        plt.scatter(
            res["end_pt"][1], res["end_pt"][0],
            c="yellow", s=30, edgecolors="black", marker="s",
            label=f"{side} End" if idx == 0 else None,
            zorder=3
        )

        # 최대 직경 위치
        plt.scatter(
            res["max_x"], res["max_y"],
            c="red", s=30, edgecolors="black",
            label=f"{side} Max Diameter" if idx == 0 else None,
            zorder=4
        )

    # Apex / Branch
    plt.scatter(
        apex_pt[1], apex_pt[0],
        c="green", s=30, edgecolors="black" ,
        label="Apex", zorder=5
    )

    if used_branch_pt:
        plt.scatter(
            used_branch_pt[1], used_branch_pt[0],
            c="green", s=30, edgecolors="black", marker="s",
            label="Branch", zorder=5
        )

    plt.title("Curvature-based Valid Limb Range")
    plt.legend(fontsize="small", loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.show()