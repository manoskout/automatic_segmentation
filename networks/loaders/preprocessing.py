import pydicom as dicom
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

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

def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":

        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val

    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor