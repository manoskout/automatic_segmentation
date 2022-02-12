import pydicom as dicom
import os 
# import RT2MASK

def get_the_doubled_rt_patients():
    doubled_rts=[]
    for patient in PATIENT_LIST:
        PATIENT_PATH = os.path.join(DATASET_PATH,patient)
        for i in os.listdir(PATIENT_PATH):
            if "STR" in i:
                struct_path= os.path.join(PATIENT_PATH,i)
                if len(os.listdir(struct_path))>1:
                    doubled_rts.append(patient)
            else:
                continue
            # print(patient)
    return doubled_rts

def get_non_masked_patients():
    non_masked=[]
    for patient in PATIENT_LIST:
        PATIENT_PATH = os.path.join(DATASET_PATH,patient)
        if "MASKS" not in os.listdir(PATIENT_PATH):
            non_masked.append(patient)
    return non_masked    

# Create a folder in datasets in order to use it to train a testing model
def create_organ_dataset(dataset_path,patients, organ):
    """
    """
    patients_mask_paths = []
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
        else:
            organ_dcm_path = [os.path.join(ORGAN_MRI_PATH,img)for img in os.listdir(ORGAN_MRI_PATH)]
            organ_mask_path = [os.path.join(ORGAN_MASK_PATH,img)for img in os.listdir(ORGAN_MASK_PATH)]
            return sorted(organ_dcm_path),sorted(organ_mask_path)
    for patient in patients:
        for patient_folder in os.listdir(patient):
            if "MASK" in patient_folder:
                patients_mask_paths.append(os.path.join(patient,patient_folder))
    for mask_path in patients_mask_paths:
        for folder in os.listdir(mask_path):
            if folder == organ:
                organ_folder = os.path.join(mask_path,folder)
                for img in os.listdir(organ_folder):
                    if ".png" in img: # That means it is a mask
                        shutil.copyfile(os.path.join(organ_folder,img), os.path.join(ORGAN_MASK_PATH,img))
                        organ_mask_path.append(os.path.join(ORGAN_MASK_PATH,img))

                    elif ".dcm" in img: # That means it is a mask
                        shutil.copyfile(os.path.join(organ_folder,img), os.path.join(ORGAN_MRI_PATH,img))
                        organ_dcm_path.append(os.path.join(ORGAN_MRI_PATH,img))
    # Clean faulty files 
    # The problem is that sometimes during copying some dumb files saved (starting with .)
    for index, mask_file in enumerate(organ_mask_path):
        if os.path.basename(mask_file)[0]==".":
            os.remove(mask_file)
            del organ_mask_path[index]
    for index, dcm_file in enumerate(organ_dcm_path):
        if os.path.basename(dcm_file)[0]==".":
            of.remove(dcm_file)
            del organ_dcm_path[index]

    print(f"Number of dcm: {len(organ_dcm_path)} Number of masks: {len(organ_mask_path)}")           
    return organ_dcm_path, organ_mask_path

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()