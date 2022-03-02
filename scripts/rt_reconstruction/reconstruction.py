import pydicom as dicom
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

def get_contours(mask : np.array) -> np.array:
    contours, hierarchy = cv.findContours(
        mask,
        cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
    
    return contours
def main():
    img_path = "input.dcm"
    mask_path = "output.png"
    rt_struct_path = "struct.dcm"
    rt_struct = dicom.dcmread(rt_struct_path)
    img = dicom.dcmread(img_path).pixel_array
    mask = cv.imread(mask_path)
    mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    mask_contours = get_contours(mask)
    
    # # print(mask)
    # plt.imshow(img, cmap = "gray")
    # plt.imshow(mask, cmap="jet", alpha=0.5)
    # plt.show()


if __name__=="__main__":
    main()