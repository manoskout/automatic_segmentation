import os
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import pydicom as dicom
import cv2 as cv

def resize_with_padding(img, new_image_size=(512, 512),color=(0,0,0)):
	"""
	This function used in order to keep the geometry of the image the same during the resize method.
	"""
	if len(img.shape)==2:
		# The image is grayscaled
		# print("The image is grayscaled")
		old_image_height, old_image_width = img.shape

		# try:
		result = np.full(new_image_size, 0, dtype=img.dtype)
		# # compute center offset
		x_center = (new_image_size[0] - old_image_width) // 2
		y_center = (new_image_size[1] - old_image_height) // 2

		
	elif len(img.shape)==3:
		old_image_height, old_image_width, channels = img.shape

		# try:
		result = np.full(new_image_size, color, dtype=img.dtype)
		# # compute center offset
		x_center = (new_image_size[0] - old_image_width) // 2
		y_center = (new_image_size[1] - old_image_height) // 2

	# # copy img image into center of result image
	result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
	return result

class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=256,mode='train',augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
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
		self.image_size = (image_size,image_size)
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		filename = os.path.basename(image_path)
		GT_path = os.path.join(self.GT_paths, filename.split(".")[0] + 'mask.png')
		to_tensor = T.Compose([
			T.ToTensor(),
			T.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225]
			)
		])

		# image = Image.open(image_path)
		# To gray scale, because I am getting grayscaled output easier to use it for the dice coeff
		# Check if it is right
		# image = ImageOps.grayscale(image)
		image_file = dicom.dcmread(image_path)
		image = image_file.pixel_array
		# print(image.dtype)
		GT = cv.imread(GT_path, cv.IMREAD_GRAYSCALE)
		# print(GT.dtype)
		image = image/np.max(image)
		image = image.astype(np.float32)
		GT = GT/np.max(GT)
		GT = GT > 0.5
		GT = GT.astype(np.float32)		
		if self.mode == "test":
			image = resize_with_padding(image, new_image_size=self.image_size)
			GT =resize_with_padding(GT, new_image_size=self.image_size)
		else:
			image = cv.resize(image,self.image_size)
			GT = cv.resize(GT,self.image_size)

		
		image = np.expand_dims(image, axis=-1)
		GT = np.expand_dims(GT,axis=-1)
		image = to_tensor(image)
		GT = to_tensor(GT)

		
		return image, GT

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
	# return dataset

# dt = get_loader("C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\TETE_FEMORALE_D\\train", image_size=512, batch_size=2)
# image,mask = dt.__getitem__(140)
# import matplotlib.pyplot as plt
# fig, (ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(np.squeeze(image))
# ax2.imshow(np.squeeze(mask))
# plt.show()