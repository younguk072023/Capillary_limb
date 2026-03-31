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
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])
        
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_dir}")

        self.index = 0
        self.current_xlim = None
        self.current_ylim = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor="black")
        self.fig.canvas.manager.set_window_title("Capillary Viewer - dual leg robust")
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
                f"[{self.index + 1}/{len(self.image_files)}]\n{os.path.basename(image_path)}\n\n{result['reason']}",
                transform=self.ax.transAxes, ha="center", va="center", color="red", fontsize=12
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

        self.ax.imshow(img, cmap="gray")
        overlay = np.ma.masked_where(labeled_mask == 0, labeled_mask)
        self.ax.imshow(overlay, cmap="nipy_spectral", alpha=0.25)

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

            px = [p[1] for p in res["path"]]
            py = [p[0] for p in res["path"]]
            self.ax.plot(px, py, color="white", linewidth=1.0, alpha=0.35, zorder=1)

            tx = [p[1] for p in res["trimmed_path"]]
            ty = [p[0] for p in res["trimmed_path"]]
            self.ax.plot(tx, ty, color="black", linewidth=4, zorder=2)
            self.ax.plot(tx, ty, color=line_color, linewidth=2.0, zorder=3, label=f"{side} measured")

            # 비대칭 양방향 거리(Ray Casting)를 반영하여 선 그리기 
            mx, my = res["max_x"], res["max_y"]
            pv = res["local_perp_v"]
            
            # 엔진에서 보내준 비대칭 거리
            d_pos = res["max_d_pos"]
            d_neg = res["max_d_neg"]

            # 중심점에서 한쪽은 d_pos만큼, 반대쪽은 d_neg만큼 뻗어나감
            p1_x, p1_y = mx + pv[0] * d_pos, my + pv[1] * d_pos
            p2_x, p2_y = mx - pv[0] * d_neg, my - pv[1] * d_neg

            # 측정선의 가시성을 높이기 위해 검은 테두리를 넣고 빨간 실선을 그립니다.
            self.ax.plot([p1_x, p2_x], [p1_y, p2_y], color="black", linewidth=5, zorder=5)
            self.ax.plot([p1_x, p2_x], [p1_y, p2_y], color="red", linewidth=2.5, zorder=6)
            
            # 수치 텍스트 표시 (위치는 측정선 위쪽)
            max_r_approx = (d_pos + d_neg) / 2.0
            self.ax.text(
                mx,
                my - max_r_approx - 10,
                f"{res['max_d']:.2f}",
                color="white", fontsize=9, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6)
            )

        if left_seed is not None:
            self.ax.scatter(left_seed[1], left_seed[0], c="cyan", s=70, edgecolors="black", marker="o", zorder=7)
        if right_seed is not None:
            self.ax.scatter(right_seed[1], right_seed[0], c="yellow", s=70, edgecolors="black", marker="o", zorder=7)
        if apex_cut_pt is not None:
            self.ax.scatter(apex_cut_pt[1], apex_cut_pt[0], c="magenta", s=80, edgecolors="white", marker="x", zorder=7)

        self.ax.scatter(U_xy[0], U_xy[1], c="gray", s=55, edgecolors="white", zorder=7, label="U-point")
        self.ax.scatter(D_xy[0], D_xy[1], c="red", s=65, edgecolors="white", zorder=7, label="D-point")

        perp_v = np.array([-dir_v[1], dir_v[0]], dtype=float)

        D_vec = np.array([D_xy[0], D_xy[1]], dtype=float)
        p1_red = D_vec + perp_v * 150
        p2_red = D_vec - perp_v * 150
        self.ax.plot([p1_red[0], p2_red[0]], [p1_red[1], p2_red[1]], color="red", linestyle="-", linewidth=2, zorder=4, label="Apex width line")

        if used_branch_pt is not None:
            B_vec = np.array([used_branch_pt[1], used_branch_pt[0]], dtype=float)
            self.ax.scatter(B_vec[0], B_vec[1], c="lime", s=70, edgecolors="black", marker="s", zorder=7)
            p1_blue = B_vec + perp_v * 150
            p2_blue = B_vec - perp_v * 150
            self.ax.plot([p1_blue[0], p2_blue[0]], [p1_blue[1], p2_blue[1]], color="blue", linestyle="-", linewidth=2, zorder=4, label="Bottom cutoff")

        title_main = f"[{self.index + 1}/{len(self.image_files)}] {search_name}"
        title_sub = " | ".join(result_text)
        self.ax.set_title(f"{title_main}\n{title_sub}", color="white", fontsize=12, pad=14)
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