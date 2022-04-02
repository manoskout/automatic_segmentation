# # Creating Train / Val / Test folders (One time use)
from doctest import REPORT_ONLY_FIRST_FAILURE
from logging import root
import os
import numpy as np
import shutil
import random
from data_aug import augment_data
import uuid

# classes_dir = ['good', 'bad'] #total labels


def both_shuffling(patient_name, image_type= ".png",mask_type=".png",):
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
def get_file_names(filenames, train, val, test):
    train_names= [image for image in filenames if image[0:3] in train]
    val_names= [image for image in filenames if image[0:3] in val]
    test_names = [image for image in filenames if image[0:3] in test]
    return train_names, val_names, test_names
if __name__ == "__main__":
    np.random.seed(42)
    root_dir = "C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\2_5D_multiclass_imbalanced" # data root path
    val_ratio = 0.1
    test_ratio = 0.2
    # for cls in classes_dir:
    # Check about the classes
    if not os.path.exists(root_dir +'\\train'):
        os.makedirs(root_dir +'\\train')
        # os.makedirs(root_dir +'\\validation')
        os.makedirs(root_dir +'\\test')

        

    # Creating partitions of the data after shuffeling
    mask_path= os.path.join(root_dir,"mask")
    img_path = os.path.join(root_dir,"mri")
    mask_filenames = os.listdir(mask_path)
    img_filenames = os.listdir(img_path)
    patient_numbers = list(set([patient[0:3] for patient in img_filenames]))
    # print(len(patient_numbers))
    patients_train, patients_val, patients_test = splitting(patient_numbers, val_ratio, test_ratio)

    # img_filenames,mask_filenames=both_shuffling(img_filenames,mask_filenames, image_type=".dcm",mask_type=".tiff")


    img_train, img_val, img_test = get_file_names(img_filenames, patients_train, patients_val, patients_test)
    mask_train, mask_val, mask_test = get_file_names(mask_filenames,patients_train, patients_val, patients_test)
    print(img_test[0:5])
    print(mask_test[0:5])    

    print('Total images: ', len(mask_filenames))
    print('Training: ', len(mask_train))
    print('Validation: ', len(mask_val))
    print('Testing: ', len(mask_test))
    augment_data(img_train, mask_train, img_path, mask_path, "C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\2_5D_multiclass_imbalanced\\train", output_type="numpy", augment=False)
    augment_data(img_test, mask_test, img_path, mask_path, "C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\2_5D_multiclass_imbalanced\\test", output_type="numpy", augment=False)
    augment_data(img_val, mask_val, img_path, mask_path, "C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\2_5D_multiclass_imbalanced\\validation", output_type="numpy", augment=False)

    # print("---------------------------------------------------------------------------------")
    # print("After augmentation..")
    # print('Training: ', len(os.listdir("C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\train\\image")))
    # print('Validation: ', len(os.listdir("C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\validation\\image")))
    # print('Testing: ', len(os.listdir("C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\test\\image")))