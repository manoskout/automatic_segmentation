

path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\MRI\\001_TETE_FEMORALE_D_104.dcm.png"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import channel_shuffle
# read image

def resize_keep_geometry(img, new_image_width=512, new_image_height=512,color=(0,0,0)):
    old_image_height, old_image_width, channels = img.shape
# create new image of desired size and color (blue) for padding
    new_image_width = 512
    new_image_height = 512
    # try:
    assert channels == len(color),f"The image should be RGB -> Image channels {channels}"
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    

    # # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
    return result



img = cv2.imread(path)
# view result
result = resize_keep_geometry(img)
plt.imshow(result)
plt.show()