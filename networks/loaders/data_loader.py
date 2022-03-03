from math import degrees
import warnings
warnings.filterwarnings("ignore") # temporal
import os
import numpy as np
import torch
from torch.utils import data
import albumentations as A
from torchvision import transforms as T
from torchvision.transforms import functional as F
import pydicom as dicom
import cv2 as cv
import torchio as tio
from loaders.preprocessing import crop_and_pad, limiting_filter



class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=256,mode='train',classes = None, augmentation_prob=0.4, is_multiorgan = True):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		self.is_multiorgan = is_multiorgan
		self.classes = classes
		# GT : Ground Truth
		img_path = os.path.join(root,"image")
		# self.GT_paths = root[:-1]+'_GT/'
		self.GT_paths = os.path.join(root,"mask")
		# self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_paths = list(
			map(
				lambda x: os.path.join(img_path,x),
				os.listdir(img_path)
			)
		)
		# A useless and complicated script to do some tests
		# self.image_paths =sorted(
		# 	self.image_paths, 
		# 	key= lambda s: int(
		# 		os.path.basename(s)
		# 		.split("_")[1]
		# 		.split(".")[0]
		# 	)
		# )
		self.image_size = (image_size,image_size)
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def mask_to_class(self,mask):
		for k in self.classes:
			mask[mask==k] = self.classes[k]
		# plt.imshow(mask)
		# plt.show()
		return mask
		
	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		filename = os.path.basename(image_path)
		GT_path = os.path.join(self.GT_paths, filename.split(".")[0] + 'mask.png')
		to_tensor = T.Compose([
			T.ToTensor(),
		])
		trans = A.Compose([
			A.Rotate(limit=(-15,15),p=0.5),
			# A.VerticalFlip(p=0.5), # Possibility to cause problems with the rectum and bladder
		])
		# This transform is only used for the MRI
		deformation = tio.Compose([
			tio.RandomElasticDeformation(
				p=0.5,
				num_control_points=7,  # or just 7
    			locked_borders=1,),
		])
		random_bias = tio.Compose([
			# tio.RandomElasticDeformation(
			# 	num_control_points=7,  # or just 7
    		# 	locked_borders=2,),
			
			tio.RandomBiasField(p=0.5)
		])



		image_file = dicom.dcmread(image_path)
		image = image_file.pixel_array
		image = limiting_filter(image,threshold=10,display=False)
		GT = cv.imread(GT_path, cv.IMREAD_GRAYSCALE)
		image = image/np.max(image)
		image = image.astype(np.float32)
		# Resize keeping the same geometry
		image = crop_and_pad(image,self.image_size,display=False)
		GT =crop_and_pad(GT, self.image_size)

		if not self.is_multiorgan:
			GT = GT/np.max(GT)
			GT = GT > 0.5
			GT = GT.astype(np.float32)	
		else:
			GT = self.mask_to_class(GT)

		# if self.mode == "test":
		

		
		image = np.expand_dims(image, axis=-1)
		GT = np.expand_dims(GT,axis=-1)
		augmented={}
		if self.mode == "train": 
			augmented = trans(image=image, mask=GT)
		else:
			augmented["image"] = image
			augmented["mask"] = GT
		
		# I used expand dims because the tranformation in torch io applied in 3D volumes
		image = random_bias(np.expand_dims(augmented["image"], axis=0))
		image = deformation(image)
		GT = deformation(np.expand_dims(augmented["mask"], axis=0))

		image = to_tensor(image.squeeze(axis=-1)).permute(1,2,0)
		GT = to_tensor(GT.squeeze(axis=-1)).permute(1,2,0)
				
		return image, GT.type(torch.long)

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',is_multiorgan=True, augmentation_prob=0.4, classes = None):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob, is_multiorgan=is_multiorgan, classes=classes)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
# 	return dataset

# import matplotlib.pyplot as plt 
# path='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\train'
# classes = {255: 1, 127: 2, 85: 3, 63: 4}
# dataload = get_loader(path,256,1,is_multiorgan=True,mode="train",classes =classes)

# # image,mask = dataload.__getitem__(55)
# for (image,mask) in dataload:
# 	# transforms = T.Compose([T.ToPILImage()])
# 	image = image.data.cpu().detach().numpy().squeeze()
# 	mask = mask.data.cpu().detach().numpy().squeeze()
# 	# print (np.unique(mask))
# 	fig, ax1 = plt.subplots(1,1)
# 	ax1.imshow(image, cmap="gray")
# 	ax1.imshow(mask, cmap="jet", alpha= 0.1 )

# 	plt.show()

