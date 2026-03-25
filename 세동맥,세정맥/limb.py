import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve, distance_transform_edt


def skeletonize_image(image,CSV_FILE):

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    #skeletonzie
    binary_image = img > 127
    skeleton = skeletonize(binary_image)

    #분할 영역 다른 미세 노이즈들은 삭제
    labeled_full = label(skeleton)
    largest_idx = np.argmax(np.bincount(labeled_full.flat)[1:])+1
    skeleton = (labeled_full ==largest_idx)
    
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
        plt.scatter(smx, smy, c="black", s=30, label="Apex")
        plt.title("Overlay with Points Image")

        # 커널을 이용한 교차점 bracnh 점 추출
        kernel = np.array([[1,1,1],
                           [1,10,1],
                           [1,1,1]])
        
        filtered = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval = 0)

        branch_ys, branch_xs = np.where(filtered >= 13)

        if len(branch_xs) > 0:
            plt.scatter(branch_xs, branch_ys, c="blue", s=30, label="branch")
            
            # 가장 가까운 첫번째 교차점 찾기(여러 교차점이 있을 수 있음을 방지)
            branch_points = np.column_stack((branch_xs, branch_ys)) 
            dist_to_apex = np.linalg.norm(branch_points - np.array([smx,smy]), axis=1)
            first_branch_idx = np.argmin(dist_to_apex)
            bx, by = branch_points[first_branch_idx]

            cut_mask = np.zeros_like(skeleton, dtype=np.uint8)
            cut_mask[smy, smx] = 1
            cut_mask[branch_ys, branch_xs] = 1
            
            min_y, max_y = min(smy, by), max(smy, by)
            mode = "Branch Mode"
        else:
            cut_mask[smy, smx] = 1
            min_y, max_y = smy, img.shape[0]
            bx,by = smx, smy
            mode = "U-shape Mode"
    

        cut_mask = cv2.dilate(cut_mask, np.ones((5, 5), np.uint8))
        cut_skeleton = skeleton.copy()
        cut_skeleton[cut_mask>0] = False

        labeled_skel = label(cut_skeleton)

        _, indices = distance_transform_edt(labeled_skel == 0, return_indices=True)
        
        labeled_mask = labeled_skel[indices[0], indices[1]]
        labeled_mask[~binary_image] = 0

        dist_map = distance_transform_edt(binary_image)

        leg_results=[]  #세동맥 세정맥의 정보 변수
        
        for i in range(1, labeled_skel.max()+1):
            comp_ys, comp_xs = np.where(labeled_skel == i)
            #조건 검사: 꼭대기와 분기점 사이가 맞는지 확인
            if len(comp_ys) > 0:
                mean_y = np.mean(comp_ys)
                if min_y < mean_y < max_y:
                    #최대 직경 찾기
                    bone_radii = dist_map[comp_ys,comp_xs]
                    max_r_idx = np.argmax(bone_radii)
                    max_d = bone_radii[max_r_idx] * 2
                 
                    leg_results.append({
                        'id': i,
                        'max_d': max_d,
                        'max_x': comp_xs[max_r_idx],
                        'max_y': comp_ys[max_r_idx],
                        'mean_x': np.mean(comp_xs)
                    })

        leg_results.sort(key=lambda x: x['mean_x'])

        print(f"\n{'='*30}")
        print(f"이미지 파일: {search_name}")
        for idx, res in enumerate(leg_results):
            side = "세동맥(Arterial)" if idx == 0 else "세정맥(Venous)"
            # res['id']는 원본 레이블 번호, res['max_d']는 계산된 최대 직경입니다.
            print(f"- {side} 최대 직경: {res['max_d']:.2f} px")
        print(f"{'='*30}\n")


        plt.subplot(2,2,4)
        plt.imshow(labeled_mask, cmap='nipy_spectral')
        for idx, res in enumerate(leg_results):
            side = "Arterial" if idx == 0 else "Venous"
            # 가장 두꺼운 지점에 빨간 별표 표시
            plt.scatter(res['max_x'], res['max_y'], c='red', s=30, label=f"{side} Max D: {res['max_d']:.2f}px")

        plt.scatter(smx,smy, c="white", s= 30, label="cut apex")
        if len(branch_xs) > 0:
            plt.scatter(bx,by, c= "white", s= 30, label="cut branch")
            
        plt.title(f"Separated ({mode})")
        plt.legend()
        plt.tight_layout()
        plt.show()

CSV_FILE = "capillary_keypoint_final.csv"
skeletonize_image("p9_det_019_crop_3.tif",CSV_FILE)

    