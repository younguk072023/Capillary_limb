import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.morphology import skeletonize

def skeletonize_image(image,CSV_FILE):

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    #skeletonzie
    binary_image = img > 127
    skeleton = skeletonize(binary_image)
    # original + skeletonize 
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[skeleton] = [0,0,255]

    #csv 검출
    df = pd.read_csv(CSV_FILE)
    search_name = os.path.splitext(os.path.basename(image))[0]
    df['pure_name']=df['filename'].apply(lambda x: os.path.splitext(x)[0])
    row = df[df['pure_name'] == search_name]

    plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    plt.imshow(img,cmap='gray')
    plt.title("Original Image")

    plt.subplot(2,2,2)
    plt.imshow(skeleton,cmap='gray')
    plt.title("Skeletonized Image")

    plt.subplot(2,2,3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    if not row.empty:
        ux, uy = row['U_x'].values[0], row['U_y'].values[0]
        dx, dy = row['D_x'].values[0], row['D_y'].values[0]

        plt.plot([ux,dx], [uy,dy], color='blue', linestyle='--', linewidth=1,label='Guide Line')

        ys, xs= np.where(skeleton>0)
        skeleton_points = np.column_stack((xs,ys))

        mid_x, mid_y = (ux+dx) /2, (uy+dy)/2

        distances = np.linalg.norm(skeleton_points - np.array([mid_x, mid_y]), axis=1)
        closet_idx = np.argmin(distances)
        smx, smy = skeleton_points[closet_idx]

        plt.scatter(ux, uy, c="Green", s=30)
        plt.scatter(dx, dy, c="Yellow", s=30)
        plt.scatter(smx, smy, c="black", s=30)
        plt.title("Overlay with Points Image")

    plt.tight_layout()
    plt.show()

CSV_FILE = "capillary_keypoint_final.csv"
skeletonize_image("p9_det_019_crop_3.tif",CSV_FILE)

    