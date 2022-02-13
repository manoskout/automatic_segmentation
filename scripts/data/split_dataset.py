# # Creating Train / Val / Test folders (One time use)
from doctest import REPORT_ONLY_FIRST_FAILURE
from logging import root
import os
import numpy as np
import shutil
import random
from data_aug import augment_data
import uuid
root_dir = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D" # data root path
organ = "TETE_FEMORALE_D"
# classes_dir = ['good', 'bad'] #total labels


def both_shuffling(image_files, mask_files, image_type= "dcm.png",mask_type="mask.png",):
    img = sorted([i for i in image_files if image_type in i and "."!=i[0]])
    masks = sorted([i for i in mask_files if mask_type in i and "."!=i[0]])
    combined = list(zip(img,masks))
    np.random.shuffle(combined)
    image_files,mask_files= zip(*combined)
    return list(image_files), list(mask_files)

def splitting(filenames,val_ratio, test_ratio):
    train, val, test = np.split(np.array(filenames),
                                                          [int(len(filenames)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(filenames)* (1 - test_ratio))])
    return train,val,test

if __name__ == "__main__":
    np.random.seed(42)
    val_ratio = 0.15
    test_ratio = 0.05
    # for cls in classes_dir:
    # Check about the classes
    if not os.path.exists(root_dir +'\\train'):
        os.makedirs(root_dir +'\\train')
        os.makedirs(root_dir +'\\validation')
        os.makedirs(root_dir +'\\test')

        

    # Creating partitions of the data after shuffeling
    mask_path= os.path.join(root_dir,"MASK")
    img_path = os.path.join(root_dir,"MRI")
    mask_filenames = os.listdir(mask_path)
    img_filenames = os.listdir(img_path)
    img_filenames,mask_filenames=both_shuffling(img_filenames,mask_filenames, image_type=".dcm",mask_type="mask.png")


    img_train, img_val, img_test = splitting(img_filenames,val_ratio, test_ratio)
    mask_train, mask_val, mask_test = splitting(mask_filenames,val_ratio, test_ratio)

    print('Total images: ', len(mask_filenames))
    print('Training: ', len(mask_train))
    print('Validation: ', len(mask_val))
    print('Testing: ', len(mask_test))
    augment_data(img_train, mask_train, img_path, mask_path, "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\train", output_type="numpy", size= (512,512), augment=False)
    augment_data(img_test, mask_test, img_path, mask_path, "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\test", output_type="numpy", size= (512,512), augment=False)
    augment_data(img_val, mask_val, img_path, mask_path, "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\validation", output_type="numpy", size= (512,512), augment=False)

    print("---------------------------------------------------------------------------------")
    print("After augmentation..")
    print('Training: ', len(os.listdir("C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\train\\image")))
    print('Validation: ', len(os.listdir("C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\validation\\image")))
    print('Testing: ', len(os.listdir("C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\test\\image")))