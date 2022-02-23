# import os
# from rt_utils import RTStructBuilder
import shutil
import matplotlib.pyplot as plt
# %matplotlib widget
import logging
logging.basicConfig(level=logging.DEBUG)
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
    print(dicom_series_path)
    for root, _, files in os.walk(dicom_series_path):

      for file in sorted(files):
        try:
          # print(file)
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
  def __init__(self, dataset_path: str, series_data: list, rt_struct: FileDataset, is_multiorgan=True, OARS = [], ROIGenerationAlgorithm=0):
    self.dataset_path = dataset_path
    self.is_multiorgan = is_multiorgan
    self.OARS = OARS
    self.series_data = series_data
    self.rt_struct = rt_struct
    self.mask_data={}
  
  def get_roi_names(self):
    """Returns a list of the names of all ROI within the RTStruct
    """
    if not self.rt_struct:
      print(self.rt_struct)
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
    organ_ids=[]
    if self.is_multiorgan:
        for seg in self.rt_struct.StructureSetROISequence:
            for oar in self.OARS:
                if oar == seg.ROIName:
                    organ_ids.append(int(seg.ROINumber)-1)
    else:
        for seg in self.rt_struct.StructureSetROISequence:
            if segment == seg.ROIName:
                return int(seg.ROINumber)-1 # Because it starts from 0
    return organ_ids

  def save_as_nifti(self,array,patient, path,affine= np.eye(4)):
    res = nib.Nifti1Image(array, affine)
    print(os.path.join(path, f'{patient}.nii.gz'))
    nib.save(res, os.path.join(path, f'{patient}.nii.gz'))
  
  def get_json_data(self,):
    """
    """
    return self.mask_data

  def clean_mask_data(self,):
    """"""
    self.mask_data={}
  
  def create_masks(self): # just hardcoded to check the rois , TODO : Add two different one for just an organ and other for multiple organs
    refROInumbers= self.get_ref_ROI_num(self.OARS) 
    # print(f"Roi Numbers : {refROInumbers}")
    slices_ms = [] 
    mask_dict = {}
    
    for index,slice in enumerate(self.series_data,1):
        # if index == :
        width, height = self.get_slice_shape(slice)
        mask_dict[slice.SOPInstanceUID] = []
        self.mask_data[str(index)]={}
        if self.rt_struct:
          for label,oar in enumerate(refROInumbers,1):
              for contour in self.rt_struct.ROIContourSequence[oar].ContourSequence:
                  if hasattr(contour,"ContourImageSequence"):
                      if contour.ContourImageSequence[0].ReferencedSOPInstanceUID == slice.SOPInstanceUID:
                          mask_coords = self.update_pixel_coords(slice, contour)
                          mask = poly_to_mask(mask_coords, width=width,height=height,label = label)
                          mask_dict[slice.SOPInstanceUID].append(mask)
                      else:
                          mask = np.zeros((height,width))
                          mask_dict[slice.SOPInstanceUID].append(mask)
          self.mask_data[str(index)]["mask"] = mask_dict[slice.SOPInstanceUID]
        self.mask_data[str(index)]["SOPInstanceUID"]= slice.SOPInstanceUID
        self.mask_data[str(index)]["slice"] = slice


  def save_masks(self,patient_path,patient):
    result_path = os.path.join(patient_path,"nifti")
    avg_value_of_classes = 255/len(self.OARS)
    if not os.path.exists(result_path):
      os.mkdir(result_path)
    masks=[]
    slices = []
    for (index,case) in self.mask_data.items():
      mask= case["mask"] if "mask" in case.keys() and self.rt_struct else None
      # print(slice)
      if mask is not None:
        mask = sum(mask) if mask is not None else None
        masks.append(mask) 

      slice = case["slice"].pixel_array
      slices.append(slice)
      
      # else:
        # mask =np.array(masks, dtype=np.uint8)
      # if mask.size!=0:
        
        # mask = Image.fromarray((mask*avg_value_of_classes).astype(np.uint8))
      # print(np.unique(mask))
        # slc = case[1]["slice"]
        # mask.save("{}/{}_{}.png".format(mask_path,patient[0:3],case[0]))
        # dicom.dcmwrite("{}/{}_{}.dcm".format(mri_path,patient[0:3],case[0]), case[1]["slice"])
        # print("Mask saved : {}/{}.png".format(os.path.join(mask_path,segment),case[0]))

    # if masks:
    # print(patient)
    # array,patient, path,affine= np.eye(4)
    self.save_as_nifti(
      np.array(masks, dtype=np.uint8).transpose([2,1,0]),
      patient+"_masks",
      result_path
      )
    self.save_as_nifti(
      np.array(slices).transpose([2,1,0]),
      patient,
      result_path
      )

    print("All files saved succesfully.")
    
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
  
  PATIENT_FOLDERS = [patient for patient in get_patient_folder_list(DATASET_PATH) if "MASKS" not in os.listdir(patient)]
  doubled_rt_structs=[]
  patient_with_doubled_rt = []
  OARS = config.oars
  logging.info(f"Selected ROIS for extraction: {OARS}") if config.include_rt_struct else logging.info("No segments for extraction")
  
  for patient_path in PATIENT_FOLDERS:
    patient = os.path.basename(patient_path)
    logging.info("Mask extraction for the patient: {} started".format(patient))
    mris = [out for out in os.listdir(patient_path) if "MR" in out ][0]
    mri_path = os.path.join(patient_path,mris)
    
    if config.include_rt_struct:
      structs = [out for out in os.listdir(patient_path) if "STRU" in out ][0]
      struct_path = os.path.join(patient_path,structs)
    else:
      struct_path = []

    output_path = config.output_path if config.output_path != " " else patient_path
    series,rt_struct = load_dcm_rt_from_path(mri_path,struct_path)
    
    mb = MaskBuilder(DATASET_PATH,series, rt_struct, is_multiorgan = True, OARS = OARS)
    roi_names = mb.get_roi_names()
    logging.info(f"Available rois: {roi_names}") if config.include_rt_struct else print("\n")
    mb.create_masks()
    mb.save_masks(patient_path, patient)
    mb.clean_mask_data()
    print("----- For the patient {}".format(patient))
    print("--------------------------------NEXT-------------------------------------")  
  print("--->> Patients that are not yet investigated because of multiple rt_struct: {} \nStruct Path: ".format(patient_with_doubled_rt,doubled_rt_structs)) 
    

if __name__ =="__main__":
  parser = argparse.ArgumentParser(description="Automated conversion from dicom series and rt struct to nifti.")

  parser.add_argument("-d","--dataset-folder", type=str, default="/Users/manoskoutoulakis/Desktop/Sample", help="The folder that contains all patients")
  parser.add_argument("--oars", nargs="+", default=["RECTUM","VESSIE","TETE_FEMORALE_D","TETE_FEMORALE_G"], help="Provide the list with the organs that you want to segment, if the names are known")
  parser.add_argument("--include_rt_struct", action="store_true", help="Create and nii file with mask. Only if the patient contains rt struct file")
  parser.add_argument("-o", "--output_path", type=str, default= " ", help="The output file is save into the patients folder by default")

  arguments= parser.parse_args()
  main(arguments)

