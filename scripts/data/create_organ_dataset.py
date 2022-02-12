import os
import shutil

from scipy.fft import dst

"""This python script created a dataset according to the organ
according to the organ that user specifies"""
import pydicom as dicom

import png 
DATASET_PATH = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\"
PATIENTS_PATHS = [os.path.join(DATASET_PATH,i) for i in os.listdir(DATASET_PATH)]
PATIENTS_MASK_PATHS = []
ORGAN = "TETE_FEMORALE_D"
save_as_dicom = True
def mri_to_png(mri_file, png_file):
    """ Function to convert from a DICOM image to png
        @param mri_file: An opened file like object to read te dicom data
        @param png_file: An opened file like object to write the png data
    """

    # Extracting data from the mri file
    plan = dicom.read_file(mri_file)
    shape = plan.pixel_array.shape

    image_2d = []
    max_val = 0
    for row in plan.pixel_array:
        pixels = []
        for col in row:
            pixels.append(col)
            if col > max_val: max_val = col
        image_2d.append(pixels)

    # Rescaling grey scale between 0-255
    image_2d_scaled = []
    for row in image_2d:
        row_scaled = []
        for col in row:
            col_scaled = int((float(col) / float(max_val)) * 255.0)
            row_scaled.append(col_scaled)
        image_2d_scaled.append(row_scaled)

    # Writing the PNG file
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)

# Create a folder in datasets in order to use it to train a testing model
def create_organ_dataset(dataset_path, organ,save_as_dicom):
    """
    """
    ORGAN_PATH =os.path.join(dataset_path,organ)
    ORGAN_MASK_PATH = os.path.join(ORGAN_PATH,"MASK")
    ORGAN_MRI_PATH = os.path.join(ORGAN_PATH,"MRI")
    organ_dcm_path = []
    organ_mask_path = []
    if not os.path.exists(ORGAN_PATH):
        os.mkdir(ORGAN_PATH)

    if os.path.exists(ORGAN_PATH):
        if not os.path.exists(ORGAN_MASK_PATH) and not os.path.exists(ORGAN_MRI_PATH):
            os.mkdir(ORGAN_MASK_PATH)
            os.mkdir(ORGAN_MRI_PATH)
    for patient in PATIENTS_PATHS:
        for patient_folder in os.listdir(patient):
            if "MASK" in patient_folder:
                PATIENTS_MASK_PATHS.append(os.path.join(patient,patient_folder))
    for mask_path in PATIENTS_MASK_PATHS:
        for folder in os.listdir(mask_path):
            if folder == organ:
                organ_folder = os.path.join(mask_path,folder)
                # print(organ_folder)
                for img in os.listdir(organ_folder):
                    if ".png" in img: # That means it is a mask
                        dst_path = os.path.join(ORGAN_MASK_PATH,img)
                        dst_path = dst_path[:-4]+"_mask"+dst_path[-4:]
                        shutil.copyfile(os.path.join(organ_folder,img), dst_path)
                        organ_mask_path.append(os.path.join(ORGAN_MASK_PATH,img))

                    elif ".dcm" in img: # That means it is a mask
                        mri_file = os.path.join(organ_folder,img)
                        dst_file = os.path.join(ORGAN_MRI_PATH,img)
                        if save_as_dicom:
                            shutil.copyfile(mri_file, dst_file)
                        else:
                            mri_file = open(os.path.join(organ_folder,img), 'rb')
                            png_file = open(f"{os.path.join(ORGAN_MRI_PATH,img)}.png", 'wb')
                            mri_to_png(mri_file, png_file)
                            png_file.close()
                        organ_dcm_path.append(os.path.join(ORGAN_MRI_PATH,img))
                    
    return organ_dcm_path, organ_mask_path


                        

create_organ_dataset(DATASET_PATH,ORGAN)