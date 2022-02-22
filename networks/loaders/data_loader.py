import os
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import pydicom as dicom
import cv2 as cv
from loaders.preprocessing import crop_and_pad, limiting_filter,normalize_intensity

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
		])


		image_file = dicom.dcmread(image_path)
		image = image_file.pixel_array
		# image = limiting_filter(image,threshold=10,display=False)
		GT = cv.imread(GT_path, cv.IMREAD_GRAYSCALE)
		image = image/np.max(image)
		image = image.astype(np.float32)
		GT = GT/np.max(GT)
		GT = GT > 0.5
		GT = GT.astype(np.float32)		
		# if self.mode == "test":
		# Resize keeping the same geometry
		image = crop_and_pad(image,self.image_size,display=False)
		GT =crop_and_pad(GT, self.image_size)


		
		image = np.expand_dims(image, axis=-1)
		GT = np.expand_dims(GT,axis=-1)
		image = to_tensor(image)
		image = normalize_intensity(image, normalization="min")

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

import matplotlib.pyplot as plt

# path = "C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\RECTUM\\validation"
# dataload = get_loader(path,256,1)
# image,mask = dataload.__getitem__(40)
# transforms = T.Compose([T.ToPILImage()])
# image = image.data.cpu().detach().numpy().squeeze()
# plt.imshow(image, cmap="gray")
# plt.show()
