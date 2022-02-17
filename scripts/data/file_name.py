import os

DATASET_PATH = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\"



PATIENT_LIST = [
    patient for patient in os.listdir(DATASET_PATH) 
    if "PRO_pCT_CGFL" in patient and ".zip" not in patient]


print(PATIENT_LIST)
mask_folder=None
for i in PATIENT_LIST:
    patient_path = os.path.join(DATASET_PATH,i)

    for root, _, files in os.walk(patient_path):
        # print(root)
        if "MASKS" in root:
            mask_folder = os.path.join(patient_path,"MASKS")
            for file in files:
                file_path = os.path.join(root,file)
                organ = os.path.basename(root)
                # print(file_path)
                if organ not in os.path.basename(file_path):
                    # print(root)
                    new_filename = os.path.join(root,i[0:3]+"_"+organ+"_"+file) 
                    os.rename(file_path, new_filename)
                else:
                    continue
    try:
        if mask_folder is not None:
            os.rename(mask_folder, mask_folder+"_"+i)
    except OSError:
        print("Error : the file exists")