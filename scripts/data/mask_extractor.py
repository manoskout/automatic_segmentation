# import os
# from rt_utils import RTStructBuilder
import shutil
import matplotlib.pyplot as plt
# %matplotlib widget
import logging
logging.basicConfig(level=logging.INFO)
import nibabel as nib
import os
import pydicom as dicom
from pydicom import dcmread
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from pathlib import Path
from pydicom.dataset import FileDataset
# Error handling
from pydicom.errors import InvalidDicomError 
import nibabel as nib
import argparse
# Custom functions
# import mri_viewer


def get_patient_folder_list(dataset_path):
    return sorted([os.path.join(dataset_path,folder) for folder in os.listdir(dataset_path) if "PRO_pCT_CGFL" in folder and ".zip" not in folder])

def plot_img_mask(img_arr, msk_arr):
  """Plots the image and the exported mask
  params: img_arr: nd.array: the slice that investigated
  params: msk_arr: nd.array: the exported mask
  """
  plt.figure()
  plt.subplot(1,2,1)
  plt.imshow(img_arr, 'gray', interpolation='none')
  plt.subplot(1,2,2)
  plt.imshow(img_arr, 'gray', interpolation='none')
  plt.imshow(msk_arr, 'jet', interpolation='none', alpha=0.5)
  plt.show()

def load_dcm_rt_from_path(dicom_series_path: str, rt_struct_path: str):
    series_data = []
    for root, _, files in os.walk(dicom_series_path):

      for file in sorted(files):
        try:
          ds = dcmread(os.path.join(root, file))
          if hasattr(ds, "pixel_array"):
              series_data.append(ds)
        except Exception:
          # Not a valid DICOM file
          continue
    if rt_struct_path:
      for root, _, files in os.walk(rt_struct_path):
        if len(files)>1:
          #  TODO : For this situation all of the RT structures that we have works only for the 1st RT struct
          # Check if you use this code for other implementation 
          logging.warning(f"WARNING -- There are more than one RT structures.We consider {files[0]} as the main ")
          rt = dcmread(os.path.join(root,files[0]))
        else:
          rt = dcmread(os.path.join(root,files[0]))
    else:
      return series_data,[]
    return series_data,rt


def poly_to_mask(polygon, width, height, label = 1):    
    """Convert polygon to mask
    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=label, fill=label)
    mask = np.array(img).astype(np.uint16)
    return mask


class MaskBuilder:
  """Wrapper class to facilitate appending and extracting ROI's within an RTStruct
  """
  def __init__(self, dataset_path: str, series_data: list, rt_struct: FileDataset, is_multiorgan=True,contours_only=False, OARS: list = [], step_splitter: int=0, ROIGenerationAlgorithm=0):
    self.dataset_path = dataset_path
    self.is_multiorgan = is_multiorgan
    self.OARS = OARS
    self.series_data = series_data
    self.rt_struct = rt_struct
    self.mask_data={}
    self.contours_only = contours_only
    self.step_splitter = step_splitter
  
  def get_roi_names(self):
    """Returns a list of the names of all ROI within the RTStruct
    """
    if not self.rt_struct:
      return []
    elif not hasattr(self.rt_struct, "StructureSetROISequence"):
      return []

    return [
      structure_roi.ROIName for structure_roi in self.rt_struct.StructureSetROISequence
    ]
  
  def get_slice_shape(self, slice):
    """Returns the shape of the selected slice
    param: slice: nd.array: the selected slice
    return: tuple: the shape of the slice (width,height)
    """
    return slice.pixel_array.shape[1], slice.pixel_array.shape[0] # width, height
  
  def get_pixel_spacing(self, slice):
    """Returns the physical distance between the center of each pixel
    param: slice: nd.array: the selected slice
    return:
      float: x_spacing
      float: y_spacing
    """
    return float(slice.PixelSpacing[0]), float(slice.PixelSpacing[1])

  def get_the_origin(self, slice):
    """Returns the center of the upper left voxel
    param: slice: nd.array: the selected slice
    return: 
      float x,
      float y
    """
    x,y,_ =slice.ImagePositionPatient
    return x,y

  def contour_to_numpy(self, contour):
    """
    """
    return np.array([contour])[0].reshape((-1,3))
  
  def update_pixel_coords(self, slice,contour):
    """
    """
    mask_array= self.contour_to_numpy(contour.ContourData)
    origin_x, origin_y = self.get_the_origin(slice)
    x_spacing, y_spacing = self.get_pixel_spacing(slice)
    return [
      (
        np.ceil((x - origin_x) / x_spacing), 
        np.ceil((y - origin_y) / y_spacing)
      )
    for x, y, _ in mask_array
    ]


  def get_ref_ROI_num(self, segment=""):
    """Return the number of the segment that we want to extract
    """
    organ_ids={}
    if self.is_multiorgan:
        for seg in self.rt_struct.StructureSetROISequence:
            for label,oar in enumerate(self.OARS,1):
                if oar == seg.ROIName.upper():
                    organ_ids[oar]= {"id":int(seg.ROINumber)-1, "label":label}
    else:
        for seg in self.rt_struct.StructureSetROISequence:
            if segment == seg.ROIName:
              organ_ids["oar"]= oar
              organ_ids["label"]= 1
              organ_ids["ROINumber"] = int(seg.ROINumber)-1
    return organ_ids

  def save_as_nifti(self,array,patient, path,affine= np.eye(4)):
    """Save the array of images to a nifti.gz file"""
    res = nib.Nifti1Image(array, affine)
    # print(os.path.join(path, f'{patient}.nii.gz'))
    nib.save(res, os.path.join(path, f'{patient}.nii.gz'))
  
  def save_as_tiff(self,array,patient,index, path):
    """Save the array of images to a nifti.gz file"""
    im = Image.fromarray(array)
    im.save("{}/{}_{}.tiff".format(
        path,
        patient,
        index)
    )
  def save_as_dicom(self,slice,patient, index, path):
    """Save the array of images to a nifti.gz file"""
    dicom.dcmwrite(
      "{}/{}_{}.dcm".format(
        path,
        patient,
        index), 
      slice
    )
  
  def clean_mask_data(self,):
    """
    If the patients are more than one it just clean the mask data 
    dictionary to export the next patient's MRIS
    """
    self.mask_data={}
  
  def create_masks(self):
    refROInumbers= self.get_ref_ROI_num(self.OARS) 
    slices_ms = [] 
    mask_dict = {}
    print(refROInumbers.items())
    for index,slice in enumerate(self.series_data,1):
        width, height = self.get_slice_shape(slice)
        mask_dict[slice.SOPInstanceUID] = []
        self.mask_data[str(index)]={}
        if self.rt_struct:
          for name, oar in refROInumbers.items():
              for contour in self.rt_struct.ROIContourSequence[oar["id"]].ContourSequence:
                  # print(contour)
                  if hasattr(contour,"ContourImageSequence"):
                      if contour.ContourImageSequence[0].ReferencedSOPInstanceUID == slice.SOPInstanceUID:                         
                          mask_coords = self.update_pixel_coords(slice, contour)
                          mask = poly_to_mask(mask_coords, width=width,height=height,label = oar["label"])
                          mask_dict[slice.SOPInstanceUID].append(mask) 
                      # HUGE MEMORY CONSUMPTION BECAUSE IT CREATES USELESS MASKS   
                      # else:
                          # mask = np.zeros((height,width))
                          # mask_dict[slice.SOPInstanceUID].append(mask) if not self.contours_only else None
                  else:
                    print("Error : 11110 ... Really bad")
          self.mask_data[str(index)]["mask"] = mask_dict[slice.SOPInstanceUID]
        if mask_dict[slice.SOPInstanceUID]:
          self.mask_data[str(index)]["SOPInstanceUID"]= slice.SOPInstanceUID
          self.mask_data[str(index)]["slice"] = slice
        else:
          del self.mask_data[str(index)]
          


  def save_masks(self,patient_path,patient, save_type,result_path):
    # avg_value_of_classes = 255/len(self.OARS)
    mri_path = os.path.join(result_path,"mri")
    mask_path = os.path.join(result_path,"mask")
    if not os.path.exists(result_path):
      os.mkdir(result_path)
    if not os.path.exists(mri_path):  
      os.mkdir(mri_path)
      os.mkdir(mask_path)

    masks=[]
    slices = []
    split_counter = 0
    for num, (index,case) in enumerate(self.mask_data.items(),1):
      # print(num)
      mask= case["mask"] if "mask" in case.keys() and self.rt_struct else None
      if mask is not None:# and len(np.unique(mask))==len(self.OARS)+1: # +1 is because se background
        split_counter += 1
        mask = sum(mask) if mask is not None else None
        mask = np.where(mask==0, mask, 255/mask).astype(np.uint8) if mask is not None else None
        masks.append(mask) 
        slice = case["slice"]
        slices.append(slice.pixel_array)
        if self.step_splitter is not None and split_counter%self.step_splitter==0:
          if save_type == "nifti":
            try:
              self.save_as_nifti(
                np.array(masks, dtype=np.uint8).transpose([2,1,0]),
                patient+ str(split_counter)  +"_masks",
                mask_path
                )
              self.save_as_nifti(
                np.array(slices).transpose([2,1,0]),
                patient+ str(split_counter),
                mri_path
                )
              # print("All files saved succesfully.")
            except ValueError:
              print("Saved in : ", result_path)

              pass
          masks=[]
          slices = []
        if save_type == "dicom":
          self.save_as_dicom(
            slice,
            patient, 
            index,
            mri_path
          )
          self.save_as_tiff(
            mask,
            patient+"_mask",
            index,
            mask_path
          )
      
      

    
    
def clean_existed_masks(dataset_path, mask_folder_name):
  PATIENT_FOLDERS = get_patient_folder_list(dataset_path)
  # Delete all the pre existed segments
  for patient_path in PATIENT_FOLDERS:
    prev_masks = (out for out in os.listdir(patient_path) if mask_folder_name in out)
    for masks in prev_masks:
      shutil.rmtree(os.path.join(patient_path,masks))
      logging.info("[Delete] patient mask : {} deleted".format(masks))

def main(config):

  print(config)
  DATASET_PATH = config.dataset_folder
  if config.delete_nifti:
    clean_existed_masks(DATASET_PATH,"nifti")
    return 0
  PATIENT_FOLDERS = [patient for patient in get_patient_folder_list(DATASET_PATH)]
  doubled_rt_structs=[]
  patient_with_doubled_rt = []
  OARS = config.oars
  logging.info(f"Selected ROIS for extraction: {OARS}") if config.include_rt_struct else logging.info("No segments for extraction")
  for patient_path in PATIENT_FOLDERS:
    patient = os.path.basename(patient_path)
    logging.info("Mask extraction for the patient: {} started".format(patient))
    mri = [out for out in os.listdir(patient_path) if "MR" in out and out[0]!="." ][0]
    mri_path = os.path.join(patient_path,mri)
    
    if config.include_rt_struct:
      struct = [out for out in os.listdir(patient_path) if "STRU" in out ][0]
      
      struct_path = os.path.join(patient_path,struct) 
    else:
      struct_path = []

    output_path = config.output_path if config.output_path != " " else patient_path
    if not os.path.exists(output_path):
      os.mkdir(output_path)
    series,rt_struct = load_dcm_rt_from_path(mri_path,struct_path)
    mb = MaskBuilder(DATASET_PATH,series, rt_struct, OARS = OARS, contours_only= config.contours_only, step_splitter=config.step_splitter)
    roi_names = mb.get_roi_names()
    logging.info(f"Available rois: {roi_names}") if config.include_rt_struct else print("\n")
    mb.create_masks()
    mb.save_masks(patient_path, patient, config.save_type, output_path)
    mb.clean_mask_data()
    print("--------------------------------NEXT-------------------------------------")  
  print("--->> Patients that are not yet investigated because of multiple rt_struct: {} \nStruct Path: ".format(patient_with_doubled_rt,doubled_rt_structs)) 
    

if __name__ =="__main__":
  parser = argparse.ArgumentParser(description="Automated conversion from dicom series and rt struct to nifti.")
  # "RECTUM","VESSIE","TETE_FEMORALE_D","TETE_FEMORALE_G"
  # parser.add_argument("-d","--dataset-folder", type=str, default="C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset", help="The folder that contains all patients")
  parser.add_argument("--oars", nargs="+", default=["RECTUM","VESSIE","TETE_FEMORALE_D","TETE_FEMORALE_G"], help="Provide the list with the organs that you want to segment, if the names are known")
  parser.add_argument("--include_rt_struct", action="store_true", help="Create and nii file with mask. Only if the patient contains rt struct file")
  parser.add_argument("-o", "--output_path", type=str, default= "C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL", help="The output file is save into the patients folder by default")
  
  parser.add_argument("-d","--dataset-folder", type=str, default="C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL", help="The folder that contains all patients")

  parser.add_argument("--contours_only", action="store_true", help="Saves only the masks that contains only the slices according to the rt structure")
  parser.add_argument("--delete_nifti", action="store_true", help="Saves only the masks that contains only the slices according to the rt structure")
  parser.add_argument("--save_type", default="dicom", help="[nifti/dicom] Save the output as nifti (series of slice) or dicom (each slice seperately)")
  parser.add_argument("--step_splitter", type= int, default=1, help="Created nifti files with 3 slices per nifti (Used for 2.5D Architectures)")
  
  arguments= parser.parse_args()
  main(arguments)

