import pydicom as dicom
import os

def load_dicom(mri):
    return dicom.dcmread(mri)

def get_patient_folder_list(dataset_path):
    return sorted([os.path.join(dataset_path,folder) for folder in os.listdir(dataset_path) if "PRO_pCT_CGFL" in folder and ".zip" not in folder])


def update_attributes(prev_mri, new_mri):
    d = 1
    for p_path, path in zip(prev_mri, new_mri):
        # if d < 1:
        # p_name = os.path.basename(p_path)
        print(f"{d}File under investigation : {os.path.basename(p_path)}")
        # name = os.path.basename(path)
        for p_file, n_file in zip(os.listdir(p_path),os.listdir(path)):
            
            
            p_slice = load_dicom(os.path.join(p_path,p_file))
            n_slice = load_dicom(os.path.join(path,n_file))
            n_slice.SOPInstanceUID= p_slice.SOPInstanceUID
            n_slice.SliceLocation = p_slice.SliceLocation
            n_slice.SeriesDescription = p_slice.SeriesDescription
            n_slice.Modality = p_slice.Modality
            n_slice.save_as(os.path.join(path,p_file))
            os.remove(os.path.join(path,n_file))
        d += 1

DATASET_PATH = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset"
PATIENT_FOLDERS = get_patient_folder_list(DATASET_PATH)
# print(PATIENT_FOLDERS)
# prev_mri = [out for out in os.listdir(PATIENT_FOLDERS) if "MR" in out]
prev_mri = []
new_mri = []
for patient in PATIENT_FOLDERS:
    if "018_PRO" in patient:
        for folder in os.listdir(patient):
            if "MR" in folder:
                prev_mri.append(os.path.join(patient,folder))
            elif "ScalarVolume" in folder:
                new_mri.append(os.path.join(patient,folder))
update_attributes(prev_mri, new_mri)