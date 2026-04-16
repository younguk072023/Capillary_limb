import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import analyze_single_image


class CapillaryViewer:
    def __init__(self, image_dir, csv_path):
        self.image_dir = image_dir
        self.csv_path = csv_path

        self.df_keypoints = pd.read_csv(csv_path)

        valid_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]
        )

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_dir}")

        self.index = 0
        self.current_xlim = None
        self.current_ylim = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor="black")
        self.fig.canvas.manager.set_window_title("Capillary Viewer - limb mask debug")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.update_view(reset_zoom=True)
        plt.show()

    def get_current_image_path(self):
        return os.path.join(self.image_dir, self.image_files[self.index])

    def update_view(self, reset_zoom=False):
        self.ax.clear()
        image_path = self.get_current_image_path()
        result = analyze_single_image(image_path, self.df_keypoints)

        if not result["ok"]:
            self.ax.imshow(np.zeros((512, 512)), cmap="gray")
            self.ax.text(
                0.5, 0.5,
                f"[{self.index + 1}/{len(self.image_files)}]\n"
                f"{os.path.basename(image_path)}\n\n{result['reason']}",
                transform=self.ax.transAxes,
                ha="center", va="center",
                color="red", fontsize=12
            )
            self.ax.axis("off")
            self.fig.tight_layout()
            self.fig.canvas.draw()
            return

        img = result["img"]
        leg_results = result["leg_results"]
        labeled_mask = result["labeled_mask"]
        U_xy = result["U_xy"]
        D_xy = result["D_xy"]
        used_branch_pt = result["used_branch_pt"]
        dir_v = result["dir_v"]
        left_seed = result["left_seed"]
        right_seed = result["right_seed"]
        apex_cut_pt = result["apex_cut_pt"]
        search_name = result["search_name"]

        # 1) 원본 이미지
        self.ax.imshow(img, cmap="gray", zorder=0)

        # 2) 실제 측정에 사용된 좌/우 mask를 더 진하게 표시
        #    left_mask=1, right_mask=2
        left_overlay = np.ma.masked_where(labeled_mask != 1, labeled_mask)
        right_overlay = np.ma.masked_where(labeled_mask != 2, labeled_mask)

        # 왼쪽 다리 mask는 cyan, 오른쪽 다리 mask는 yellow 느낌으로 강하게 표시
        self.ax.imshow(left_overlay, cmap="cool", alpha=0.50, zorder=1)
        self.ax.imshow(right_overlay, cmap="Wistia", alpha=0.50, zorder=1)

        # 3) 원본 흰색 혈관 윤곽을 얇게 강조해서,
        #    실제 흰색 영역과 측정 mask 차이를 눈으로 보기 쉽게 함
        binary_edge = img > 127
        self.ax.contour(
            binary_edge.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.8,
            alpha=0.6,
            zorder=2
        )

        result_text = []
        if len(leg_results) >= 2:
            sorted_leg_results = sorted(leg_results, key=lambda x: x["max_d"], reverse=True)
            result_text.append(f"Venous: {sorted_leg_results[0]['max_d']:.2f}px")
            result_text.append(f"Arterial: {sorted_leg_results[1]['max_d']:.2f}px")
        else:
            for i, r in enumerate(leg_results):
                result_text.append(f"Leg{i+1}: {r['max_d']:.2f}px")

        for idx, res in enumerate(leg_results):
            side = "Left" if idx == 0 else "Right"
            line_color = "cyan" if idx == 0 else "yellow"

            # 전체 path
            px = [p[1] for p in res["path"]]
            py = [p[0] for p in res["path"]]
            self.ax.plot(px, py, color="white", linewidth=1.0, alpha=0.35, zorder=3)

            # 실제 측정에 사용된 trimmed path
            tx = [p[1] for p in res["trimmed_path"]]
            ty = [p[0] for p in res["trimmed_path"]]
            self.ax.plot(tx, ty, color="black", linewidth=4, zorder=4)
            self.ax.plot(tx, ty, color=line_color, linewidth=2.0, zorder=5, label=f"{side} measured")

            # 최대 직경선
            mx, my = res["max_x"], res["max_y"]
            pv = res["local_perp_v"]
            d_pos = res["max_d_pos"]
            d_neg = res["max_d_neg"]

            p1_x, p1_y = mx + pv[0] * d_pos, my + pv[1] * d_pos
            p2_x, p2_y = mx - pv[0] * d_neg, my - pv[1] * d_neg

            self.ax.plot([p1_x, p2_x], [p1_y, p2_y], color="black", linewidth=5, zorder=6)
            self.ax.plot([p1_x, p2_x], [p1_y, p2_y], color="red", linewidth=2.5, zorder=7)

            # 중심점 표시
            self.ax.scatter(mx, my, c="red", s=28, edgecolors="white", linewidths=0.8, zorder=8)

            # 수치 텍스트
            max_r_approx = (d_pos + d_neg) / 2.0
            self.ax.text(
                mx,
                my - max_r_approx - 10,
                f"{res['max_d']:.2f}",
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7),
                zorder=9
            )

        # seed / keypoint / 기준선 표시
        if left_seed is not None:
            self.ax.scatter(left_seed[1], left_seed[0], c="cyan", s=70,
                            edgecolors="black", marker="o", zorder=10)
        if right_seed is not None:
            self.ax.scatter(right_seed[1], right_seed[0], c="yellow", s=70,
                            edgecolors="black", marker="o", zorder=10)
        if apex_cut_pt is not None:
            self.ax.scatter(apex_cut_pt[1], apex_cut_pt[0], c="magenta", s=80,
                            edgecolors="white", marker="o", zorder=10)

        self.ax.scatter(U_xy[0], U_xy[1], c="gray", s=55,
                        edgecolors="white", zorder=10, label="U-point")
        self.ax.scatter(D_xy[0], D_xy[1], c="red", s=65,
                        edgecolors="white", zorder=10, label="D-point")

        perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

        D_vec = np.array([D_xy[0], D_xy[1]], dtype=float)
        p1_red = D_vec + perp_v * 150
        p2_red = D_vec - perp_v * 150
        self.ax.plot(
            [p1_red[0], p2_red[0]],
            [p1_red[1], p2_red[1]],
            color="red", linestyle="-", linewidth=2, zorder=4,
            label="Apex width line"
        )

        if used_branch_pt is not None:
            B_vec = np.array([used_branch_pt[1], used_branch_pt[0]], dtype=float)
            self.ax.scatter(B_vec[0], B_vec[1], c="lime", s=70,
                            edgecolors="black", marker="s", zorder=10)
            p1_blue = B_vec + perp_v * 150
            p2_blue = B_vec - perp_v * 150
            self.ax.plot(
                [p1_blue[0], p2_blue[0]],
                [p1_blue[1], p2_blue[1]],
                color="blue", linestyle="-", linewidth=2, zorder=4,
                label="Bottom cutoff"
            )

        title_main = f"[{self.index + 1}/{len(self.image_files)}] {search_name}"
        title_sub = " | ".join(result_text)
        title_debug = "white contour = original vessel, cyan/yellow fill = actual measurement mask"
        self.ax.set_title(
            f"{title_main}\n{title_sub}\n{title_debug}",
            color="white", fontsize=11, pad=14
        )

        self.ax.axis("off")
        self.ax.legend(fontsize="small", loc="upper right")
        self.fig.tight_layout()

        if reset_zoom or self.current_xlim is None or self.current_ylim is None:
            self.ax.set_xlim(0, img.shape[1])
            self.ax.set_ylim(img.shape[0], 0)
            self.current_xlim = self.ax.get_xlim()
            self.current_ylim = self.ax.get_ylim()
        else:
            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)

        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.image_files)
            self.current_xlim = None
            self.current_ylim = None
            self.update_view(reset_zoom=True)
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.image_files)
            self.current_xlim = None
            self.current_ylim = None
            self.update_view(reset_zoom=True)
        elif event.key == "r":
            self.current_xlim = None
            self.current_ylim = None
            self.update_view(reset_zoom=True)

    def on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1.0

        xdata = event.xdata
        ydata = event.ydata

        cur_width = (cur_xlim[1] - cur_xlim[0])
        cur_height = (cur_ylim[1] - cur_ylim[0])
        new_width = cur_width * scale_factor
        new_height = cur_height * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_width + 1e-12)
        rely = (cur_ylim[1] - ydata) / (cur_height + 1e-12)

        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()
        self.fig.canvas.draw()


if __name__ == "__main__":
    CSV_PATH = r"capillary_keypoint_final.csv"
    LABEL_DIR = r"D:\usb\MTL_dataset\label"

    if not os.path.exists(CSV_PATH):
        print(f"CSV 파일이 없습니다: {CSV_PATH}")
    elif not os.path.exists(LABEL_DIR):
        print(f"라벨 폴더가 없습니다: {LABEL_DIR}")
    else:
        CapillaryViewer(LABEL_DIR, CSV_PATH)