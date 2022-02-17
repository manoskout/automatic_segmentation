# import os
# from rt_utils import RTStructBuilder
import shutil
import matplotlib.pyplot as plt
# %matplotlib widget
import logging
logging.basicConfig(level=logging.DEBUG)

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
      for file in files:
        try:
          ds = dcmread(os.path.join(root, file))
          if hasattr(ds, "pixel_array"):
              series_data.append(ds)
        except Exception:
          # Not a valid DICOM file
          continue
    for root, _, files in os.walk(rt_struct_path):
      if len(files)>1:
        #  TODO : For this situation all of the RT structures that we have works only for the 1st RT struct
        # Check if you use this code for other implementation 
        logging.warn(f"WARNING -- There are more than one RT structures.We consider {files[0]} as the main ")
        rt = dcmread(os.path.join(root,files[0]))
      else:
        rt = dcmread(os.path.join(root,files[0]))
    # series_data = sorted(series_data, key=lambda s: s.SliceLocation)
    return series_data,rt


def poly_to_mask(polygon, width, height):
    from PIL import Image, ImageDraw
    
    """Convert polygon to mask
    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask


class MaskBuilder:
  """Wrapper class to facilitate appending and extracting ROI's within an RTStruct
  """
  def __init__(self, dataset_path: str, series_data: list, rt_struct: FileDataset, ROIGenerationAlgorithm=0):
    self.dataset_path = dataset_path
    self.series_data = series_data
    self.rt_struct = rt_struct
    self.frame_of_reference_uid = rt_struct.ReferencedFrameOfReferenceSequence[
      -1].FrameOfReferenceUID  # Use last strucitured set ROI
    self.mask_data={}
  
  def get_roi_names(self):
    """Returns a list of the names of all ROI within the RTStruct
    """
    if not self.rt_struct.StructureSetROISequence:
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


  def get_ref_ROI_num(self, segment : str):
    """Return the number of the segment that we want to extract
    """
    for seg in self.rt_struct.StructureSetROISequence:
      if segment == seg.ROIName:
        return int(seg.ROINumber)-1 # Because it starts from 0
      else:
        continue
    return []

  def concatinate_mask(self,prev_mask, mask):
    # print(prev_mask)
    # print(mask)
    return prev_mask+mask
  
  def get_json_data(self,):
    """
    """
    return self.mask_data

  def clean_mask_data(self,):
    """"""
    self.mask_data={}
  
  def create_masks(self,segment : str):
    refROInumber= self.get_ref_ROI_num(segment) 
    slices_ms = [] 
    masks = []
    indexes = []
    prev_index = 0
    for index,slice in enumerate(self.series_data,1):
      width, height = self.get_slice_shape(slice)
      
      for contour in self.rt_struct.ROIContourSequence[refROInumber].ContourSequence:
        if hasattr(contour,"ContourImageSequence"):
          if contour.ContourImageSequence[0].ReferencedSOPInstanceUID == slice.SOPInstanceUID:
            self.mask_data[str(index)]={}
            mask_coords = self.update_pixel_coords(slice,contour)
            mask = poly_to_mask(mask_coords, width, height)
            # print("Curr index: {}, prev: {}".format(index,prev_index))
            
            if index == prev_index:
              masks[-1] = self.concatinate_mask(masks[-1],mask)
              self.mask_data[str(index)]["ReferencedSegment"]= segment
              self.mask_data[str(index)]["SOPInstanceUID"]= slice.SOPInstanceUID
              self.mask_data[str(index)]["slice"]= slice

              self.mask_data[str(index)]["mask"]= masks[-1]
              prev_index = 0          
            elif index != prev_index :
            # elif prev_index is None:
              self.mask_data[str(index)]["ReferencedSegment"]= segment
              self.mask_data[str(index)]["SOPInstanceUID"]= slice.SOPInstanceUID
              self.mask_data[str(index)]["slice"]= slice
              self.mask_data[str(index)]["mask"]= mask
              
              indexes.append(index)
              slices_ms.append(slice.pixel_array)
              masks.append(mask)
              prev_mask = mask
              prev_index=index
            else:
              # print(index)
              print("ERROR 1155: It should not be shown!")
              # print(index)          
            prev_index=index
          else:
            # print(f"Segmeng: {segment} has no attrubute : ContourImageSequence. Aborted")
            continue
          
          # print("Contour with id : {} is related to image {}".format(contour.ContourImageSequence[0].ReferencedSOPInstanceUID, str(index)))
    # mri_viewer.multi_slice_viewer(masks,slices_ms, "MASK", "MRI")

  # def delete_all_masks(self,):


  def save_masks(self,patient_path, segment):
    mask_path= os.path.join(patient_path,"MASKS")
    if not os.path.exists(mask_path):
      os.mkdir(mask_path)

    if not os.path.exists(os.path.join(mask_path,segment)):
      os.mkdir(os.path.join(mask_path,segment))
    
    for case in self.mask_data.items():
      msk = Image.fromarray(case[1]["mask"])
      slc = case[1]["slice"]
      msk.save("{}/{}.png".format(os.path.join(mask_path,segment),case[0]))
      # slc.save("{}/slc{}.png".format(os.path.join(MASK_PATH,segment),case[0]))
      # print(type(case[1]["slice"]))
      dicom.dcmwrite("{}/{}.dcm".format(os.path.join(mask_path,segment),case[0]), case[1]["slice"])
      # print("Mask saved : {}/{}.png".format(os.path.join(mask_path,segment),case[0]))
    print("All files saved succesfully.")
    


    


    





# ------------- DEBUG ---------------
# series,rt_struct = load_dcm_rt_from_path(MRI_PATH,STRUCT_FILE)
# mb = MaskBuilder(series, rt_struct)
# roi_names = mb.get_roi_names()
# print(roi_names[0].isupper())
# for roi in roi_names:
#   if roi.isupper():
#     mb.create_masks(roi)# Make it automatically (using args)
#     mb.save_masks(roi)
#     mb.clean_mask_data()
def clean_existed_masks(dataset_path, mask_folder_name):
  PATIENT_FOLDERS = get_patient_folder_list(dataset_path)
  # Delete all the pre existed segments
  for patient_path in PATIENT_FOLDERS:
    prev_masks = (out for out in os.listdir(patient_path) if mask_folder_name in out)
    for masks in prev_masks:
      shutil.rmtree(os.path.join(patient_path,masks))
      logging.info("[Delete] patient mask : {} deleted".format(masks))



DATASET_PATH = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset"
PATIENT_FOLDERS = get_patient_folder_list(DATASET_PATH)
doubled_rt_structs=[]
patient_with_doubled_rt = []
for patient_path in PATIENT_FOLDERS:
  patient = os.path.basename(patient_path)
  logging.info("Mask extraction for the patient: {} started".format(patient))
  mris, structs = (out for out in os.listdir(patient_path) if "ScalarVolume" in out or "STRU" in out )
  mri_path = os.path.join(patient_path,mris)
  struct_path = os.path.join(patient_path,structs)
  series,rt_struct = load_dcm_rt_from_path(mri_path,struct_path)
  mb = MaskBuilder(DATASET_PATH,series, rt_struct)
  roi_names = mb.get_roi_names()
  for roi in roi_names:
    print(roi)
    mb.create_masks(roi)
    mb.save_masks(patient_path, roi)
    mb.clean_mask_data()
    print("----- For the patient {}".format(patient))
  print("--------------------------------NEXT-------------------------------------")  
print("--->> Patients that are not yet investigated because of multiple rt_struct: {} \nStruct Path: ".format(patient_with_doubled_rt,doubled_rt_structs)) 
    

# Patients that are not yet investigated because of multiple rt_struct: ['054_PRO_pCT_CGFL', '058_PRO_pCT_CGFL', '060_PRO_pCT_CGFL', '061_PRO_pCT_CGFL'] 

# print(mb.get_json_data())

# Sort slices according to the slice location
# slices = sorted(slices, key=lambda s: s.SliceLocation)