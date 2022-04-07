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

def augment_data(images, masks, existed_imgs, existed_masks, save_path, output_type ="numpy", augment=True):
    create_dir(os.path.join(save_path,"image"))
    create_dir(os.path.join(save_path,"mask"))

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = os.path.basename(x)

        """ Reading image and mask """
        if output_type== "numpy":
            x = os.path.join(existed_imgs,x)
            y = os.path.join(existed_masks,y)
        else:
            x = cv2.imread(os.path.join(existed_imgs,x), cv2.IMREAD_COLOR)
            # x = dicom.dcmread(x).pixel_array
            y = imageio.mimread(os.path.join(existed_masks,y))[0]

        if augment == True:
            aug = VerticalFlip(p=0.25)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=10, p=0.5)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x2, x3]
            Y = [y, y2, y3]
        else:
            X = [x]
            Y = [y]

        uid = os.path.basename(x)
        for index,(i, m) in enumerate(zip(X, Y)):
            print("image: ", os.path.basename(i), "  ,  mask", os.path.basename(m))
            if output_type == "numpy":
                tmp_image_name = f"{uid}"
                tmp_mask_name = f"{uid}"
                image_path = os.path.join(save_path, "image", tmp_image_name)
                mask_path = os.path.join(save_path, "mask", tmp_mask_name)
                shutil.copy(i, image_path)
                shutil.copy(m, mask_path)

            else:      
                tmp_image_name = f"{uid}_{index}.png"
                tmp_mask_name = f"{uid}_{index}mask.png"
                image_path = os.path.join(save_path, "image", tmp_image_name)
                mask_path = os.path.join(save_path, "mask", tmp_mask_name)
                cv2.imwrite(image_path, i)
                cv2.imwrite(mask_path, m)
