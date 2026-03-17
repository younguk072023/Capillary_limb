import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def skeletonize_image(image):

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    binary_image = img > 127

    skeleton = skeletonize(binary_image)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img,cmap='gray')
    plt.title("Original Image")

    plt.subplot(1,2,2)
    plt.imshow(skeleton,cmap='gray')
    plt.title("Skeletonized Image")


    plt.show()

skeletonize_image("p9_det_019_crop_3.tif")
    