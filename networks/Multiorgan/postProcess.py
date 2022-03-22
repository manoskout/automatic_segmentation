import os
from re import L
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# Fix memory problem
torch.cuda.empty_cache()

import matplotlib.pyplot as plt 
import cv2
def _organ_check(classes):
    assert 'VESSIE' in classes, 'VESSIE is not in classes'
    assert 'RECTUM' in classes, 'RECTUM is not in classes'
    assert 'TETE_FEMORALE_G' in classes, 'TETE_FEMORAL_G is not in classes'
    assert 'TETE_FEMORALE_D' in classes, 'TETE_FEMORAL_D is not in classes'

def vessie_check(cur_centroids, classes, contours):
    _organ_check(classes)
    cur_centroid = max(cur_centroids)
    cur_Y =cur_centroid[1]
    cur_X =cur_centroid[0]
    print("cur :", cur_centroids)
    print("rectum :", contours["RECTUM"]['centroids'])
    print("femoralD :", contours["TETE_FEMORALE_D"]['centroids'])
    print("femoralG :", contours["TETE_FEMORALE_G"]['centroids'])
    rectum_X,rectum_Y = min(contours["RECTUM"]['centroids']) if contours["RECTUM"]['centroids'] else [128,250]
    femoralD_X,femoralD_Y = max(contours["TETE_FEMORALE_D"]['centroids']) if contours["TETE_FEMORALE_D"]['centroids'] else [0,0]
    femoralG_X,femoralG_Y =  max(contours["TETE_FEMORALE_G"]['centroids']) if contours["TETE_FEMORALE_G"]['centroids'] else [0,0]
    

    try:
        assert cur_Y > rectum_Y
        assert cur_X > femoralD_X
        assert cur_X < femoralG_X
        return True
    except AssertionError:
        return False


def rules(pred_image,classes, contours):
    # contours["VESSIE"]["centroid"]
    pred_image = pred_image.squeeze()
    for seg,cl in zip(pred_image,classes):
        if cl =="BACKGROUND" or not contours[cl]['centroids']:
            continue

        if vessie_check(contours[cl]['centroids'],classes,contours) and cl=="VESSIE":
            pred_image[contours[cl]["index"],:,:] = 0 
        else:
            continue

    return pred_image

def get_contours(im):
    im = np.array(im * 255, dtype = np.uint8)
    ret,thresh = cv2.threshold(im,127,255,0)
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return contours
def get_centroids(contours):
    centroids = []
    for c in contours:
        try:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            centroids.append([cX,cY])
        except ZeroDivisionError:
            print("Zero Division")          
    return centroids
    
def plot_results(img):
    for seg in img:
        seg = seg.data.cpu().numpy()
        fig,ax = plt.subplots(1,1)
        # ax.set_title(cl)
        ax.imshow(seg, cmap="gray")
        plt.show()

def post_processing(pred_image, classes, verbose=False):
    contours = {}

    for index, (cl,segmented_class) in enumerate(zip(classes,pred_image.squeeze())):
        contours[cl]={}
        contours[cl]["index"] = index
        seg = segmented_class.data.cpu().numpy()
        if cl != "BACKGROUND":
            contours[cl]["contours"] = get_contours(seg)
            contours[cl]["centroids"] = get_centroids(contours[cl]["contours"])            
        else:
            seg =  (seg>.5)
            seg = seg == False
        
    pred_image=rules(pred_image,classes,contours)
    if verbose:
        plot_results(pred_image)
    return pred_image


classes = ["BACKGROUND","RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"]
pred_folder ='C:\\Users\\ek779475\\Documents\\Koutoulakis\\predict\\'
img_files = sorted([os.path.join(pred_folder, i) for i in os.listdir(pred_folder) if "img" in i])
pred_files = sorted([os.path.join(pred_folder, i) for i in os.listdir(pred_folder) if "img" not in i])

def classes_to_mask(classes, mask : torch.Tensor) -> torch.Tensor:
    """Converts the labeled pixels to range 0-255
    """
    for index, k in enumerate(classes):
        mask[mask==index] = int(255/index) if index != 0 else 0
    return mask.type(torch.float)

for img,pred in zip(img_files, pred_files):
    img = torch.load(img)
    image = img[0]

    pred_mask = torch.load(pred)
    real_pred = torch.load(pred)
    real_pred = real_pred.data.cpu()

    real_pred = classes_to_mask(classes,real_pred)

    pred_mask = post_processing(pred_mask, classes)
    pred_mask = classes_to_mask(classes,pred_mask)
    pred_mask = pred_mask.data.cpu()
    fig, (ax1,ax2) = plt.subplots(1,2)
    pred_mask = torch.unsqueeze(pred_mask, dim=0)
    pred_mask = torch.argmax(pred_mask,dim=1)
    real_pred = torch.argmax(real_pred,dim=1)



    pred_mask = np.ma.masked_where(pred_mask == 0, pred_mask)
    real_pred = np.ma.masked_where(real_pred == 0, real_pred)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax1.imshow(image.squeeze(),cmap="gray",interpolation='none')
    ax1.imshow(real_pred.squeeze(),cmap="cool",interpolation='none', alpha = 0.5)
    

    ax2.imshow(image.squeeze(),cmap="gray",interpolation='none')
    ax2.imshow(pred_mask.squeeze(),cmap="cool",interpolation='none', alpha = 0.5)

    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    plt.show()
    # plt.savefig(o

