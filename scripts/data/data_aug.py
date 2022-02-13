import os
import shutil
from matplotlib import image
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import pydicom as dicom
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import uuid

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "train", "*.dcm.png")))
    train_y = sorted(glob(os.path.join(path, "train", "*mask.png")))
    test_x = sorted(glob(os.path.join(path, "test", "*.dcm.png")))
    test_y = sorted(glob(os.path.join(path, "test", "*mask.png")))

    return (train_x, train_y), (test_x, test_y)

def resize_with_padding(img, new_image_size=(512, 512),color=(0,0,0)):
    """
    This function used in order to keep the geometry of the image the same during the resize method.
    """
    if len(img.shape)==2:
        # The image is grayscaled
        # print("The image is grayscaled")
        old_image_height, old_image_width = img.shape

        # try:
        # assert channels == len(color),f"The image should be RGB -> Image channels {channels}"
        result = np.full((new_image_size[0],new_image_size[1]), 1, dtype=np.uint8)
        # # compute center offset
        x_center = (new_image_size[0] - old_image_width) // 2
        y_center = (new_image_size[1] - old_image_height) // 2

        
    elif len(img.shape)==3:
        old_image_height, old_image_width, channels = img.shape

        # try:
        assert channels == len(color),f"The image should be RGB -> Image channels {channels}"
        result = np.full((new_image_size[0],new_image_size[1], channels), color, dtype=np.uint8)
        # # compute center offset
        x_center = (new_image_size[0] - old_image_width) // 2
        y_center = (new_image_size[1] - old_image_height) // 2

    # # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
    return result
def augment_data(images, masks, existed_imgs, existed_masks, save_path, size= (512,512), output_type ="numpy", augment=True):
    create_dir(os.path.join(save_path,"image"))
    create_dir(os.path.join(save_path,"mask"))

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = os.path.basename(x)

        """ Reading image and mask """
        if output_type== "numpy":
            x = os.path.join(existed_imgs,x)
            y = imageio.mimread(os.path.join(existed_masks,y))[0]
        else:
            x = cv2.imread(os.path.join(existed_imgs,x), cv2.IMREAD_COLOR)
            # x = dicom.dcmread(x).pixel_array
            y = imageio.mimread(os.path.join(existed_masks,y))[0]

        if augment == True:
            # aug = HorizontalFlip(p=1.0)
            # augmented = aug(image=x, mask=y)
            # x1 = augmented["image"]
            # y1 = augmented["mask"]

            aug = VerticalFlip(p=0.5)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=0.5)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            # X = [x, x1, x2, x3]
            # Y = [y, y1, y2, y3]
            X = [x, x2, x3]
            Y = [y, y2, y3]
        else:
            X = [x]
            Y = [y]

        index = 0
        uid =  str(uuid.uuid4())[0:12]
        for i, m in zip(X, Y):
            # i = resize_with_padding(i, size)
            # m = resize_with_padding(m, size)
            if output_type == "numpy":
                tmp_image_name = f"{uid}_{index}.dcm"
                tmp_mask_name = f"{uid}_{index}mask.png"
                image_path = os.path.join(save_path, "image", tmp_image_name)
                mask_path = os.path.join(save_path, "mask", tmp_mask_name)
                # i = np.expand_dims(i,axis=2)
                shutil.copy(i, image_path)
                cv2.imwrite(mask_path, m)
            else:      
                tmp_image_name = f"{uid}_{index}.png"
                tmp_mask_name = f"{uid}_{index}mask.png"
                image_path = os.path.join(save_path, "image", tmp_image_name)
                mask_path = os.path.join(save_path, "mask", tmp_mask_name)
                cv2.imwrite(image_path, i)
                cv2.imwrite(mask_path, m)

            index += 1

