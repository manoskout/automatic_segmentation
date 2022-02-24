import cv2 as cv
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
def res_plot(mask,contours):
    plt.subplot(121)
    plt.imshow(mask, cmap="gray", interpolation="none")
    
    plt.subplot(122)
    cv.drawContours(mask, contours, -1, (210,210,210),1)
    plt.imshow(mask, cmap="gray", interpolation="none")
    plt.show()

mask_path = "016_77.png"
mask = cv.imread(mask_path,cv.IMREAD_UNCHANGED)
classes = ["RECTUM","VESSIE","TETE_FEMORALE_D","TETE_FEMORALE_G"]
classes_pixel_values = np.array(np.flip(np.unique(mask)), dtype=np.int64)
label_contours = {}
# TODO --> The contours' structure in rt dicom file is flatten having three different values(x,y,z)
for value, label in zip(classes_pixel_values,classes):
    print(f"Label: {label}, with Pixel Values : {value}")
    tmp_mask = np.where(mask==value,mask,0)
    contours,hierarchy = cv.findContours(
        tmp_mask, 
        cv.RETR_TREE, 
        cv.CHAIN_APPROX_SIMPLE,
        offset=(0,0)
    )
    contours = contours[0].squeeze().flatten()
    # print(f"Shape of contours: {contours[0].squeeze().flatten().shape}")
    print(contours)
    label_contours[label]= contours
    plt.imshow(tmp_mask, cmap="gray")
    plt.show()

contours,hierarchy = cv.findContours(
        mask.copy(), 
        cv.RETR_TREE, 
        cv.CHAIN_APPROX_SIMPLE,
        offset=(0,0)
    )
# res_plot(mask,contours[1])
