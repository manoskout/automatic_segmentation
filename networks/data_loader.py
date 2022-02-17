import os
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import pydicom as dicom
import cv2 as cv
import matplotlib.pyplot as plt

def limiting_filter(img,threshold = 8, display=False):
    ret1,th1 = cv.threshold(img,threshold,255,cv.THRESH_BINARY)
    binary_mask = img > ret1
    output = np.zeros_like(img)
    output[binary_mask] = img[binary_mask]
    if display:
        fig , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2)
        ax1.imshow(img,cmap="gray")
        ax1.title.set_text("Original Image")
        ax2.imshow(img >0,cmap="gray")
        ax2.title.set_text("Pixels > 0")

        ax3.imshow(output, cmap="gray")
        ax3.title.set_text("Output")
        ax4.imshow(output>0,cmap="gray")
        ax4.title.set_text("After limiting Pixels > 0")
        plt.show()
    return output
def crop_and_pad(img,size, display=False):
    cropx, cropy = size
    h, w = img.shape
    starty = startx = 0
    # print(cropx, cropy, h,w)
    
    # Crop only if the crop size is smaller than image size
    if cropy <= h:   
        starty = h//2-(cropy//2)    
        
    if cropx <= w:
        startx = w//2-(cropx//2)
        
    cropped_img = img[starty:starty+cropy,startx:startx+cropx]
    # print('Cropped: ',cropped_img.shape)
    
    # Add padding, if the image is smaller than the desired dimensions
    old_image_height, old_image_width = cropped_img.shape
    new_image_height, new_image_width = old_image_height, old_image_width
    
    if old_image_height < cropy:
        new_image_height = cropy
    if old_image_width < cropy:
        new_image_width = cropy
    
    if (old_image_height != new_image_height) or (old_image_width != new_image_width):
    
        padded_img = np.full((new_image_height, new_image_width), 0, dtype=np.float32)
    
        x_center = (new_image_height - old_image_width) // 2
        y_center = (new_image_width - old_image_height) // 2
        
        padded_img[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = cropped_img
        
        # print('Padded: ',padded_img.shape)
        result = padded_img
    else:
        result = cropped_img
        
    # print('Result: ',result.shape)
        
    if display:
        plt.figure()
        plt.subplot(121, title='before cropping')
        plt.imshow(img, cmap='gray')
        plt.subplot(122, title='after cropping')
        plt.imshow(result, cmap='gray')
        # plt.subplot(133, title='resizing to original')
        # plt.imshow(resizing_func(x, image.shape[0], image.shape[1]), cmap='gray')
        plt.show()
        
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
			# T.Normalize(
			# 	mean=[0.485, 0.456, 0.406],
			# 	std=[0.229, 0.224, 0.225]
			# )
		])

		# image = Image.open(image_path)
		# To gray scale, because I am getting grayscaled output easier to use it for the dice coeff
		# Check if it is right
		# image = ImageOps.grayscale(image)
		image_file = dicom.dcmread(image_path)
		image = image_file.pixel_array
		image = limiting_filter(image,threshold=10,display=False)
		# print(image.dtype)
		GT = cv.imread(GT_path, cv.IMREAD_GRAYSCALE)
		# print(GT.dtype)
		image = image/np.max(image)
		image = image.astype(np.float32)
		GT = GT/np.max(GT)
		GT = GT > 0.5
		GT = GT.astype(np.float32)		
		# if self.mode == "test":
		# Resize keeping the same geometry
		image = crop_and_pad(image,self.image_size,display=False)
		GT =crop_and_pad(GT, self.image_size)

		

		# else:
			# image = cv.resize(image,self.image_size)
			# GT = cv.resize(GT,self.image_size)

		
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