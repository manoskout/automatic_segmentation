import os
from re import L
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
# Fix memory problem
torch.cuda.empty_cache()
from networks.network import *
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from loaders.data_loader import get_loader
from Multiorgan.utils import build_model, classes_to_mask
# from Multiorgan.rt_stuct_generator import RT_Struct
import matplotlib.pyplot as plt 
import cv2
import imutils

# def vessie_correction(img):
def post_processing(cfg,pred_image, classes):
    print(classes)
    print(pred_image.shape)
    print(pred_image.squeeze().shape)
    rectum_area=0
    vessie_area= 0
    femoral_g_area = 0
    femoral_d_area = 0


    for index,(cl,segmented_class) in enumerate(zip(classes,pred_image.squeeze())):
        # segmented_class = torch.softmax(segmented_class,dim=1)
        # if cl == "VESSIE":
            # segmented_class= vessie_correction(segmented_class)
            # segmented_class[int(cfg.image_size/2):,]
        fig,ax = plt.subplots(1,1)
        ax.set_title(cl)
        seg = segmented_class.data.cpu()
        # print(seg)
        ax.imshow(seg, cmap="gray")
        plt.show()
    return pred_image
def save_validation_results(cfg,image, pred_mask,counter=0):#pred_mask_1,pred_mask_2,pred_mask_3,counter = 0 ):
    # slices= np.arange(4,70,1)
    # if len(cfg.classes)>1:
    #     pred_mask = torch.softmax(pred_mask,dim=1)
    #     pred_mask_1 = torch.softmax(pred_mask_1, dim=1)
    #     pred_mask_2 = torch.softmax(pred_mask_2, dim=1)
    #     if counter in slices:
    #         torch.save(pred_mask, os.path.join(cfg.result_path,f"test_{counter}.pt"))
    #         torch.save(image, os.path.join(cfg.result_path,f"img_test_{counter}.pt"))

    #         return
        # pred_mask = post_processing(cfg,pred_mask, cfg.classes)

        # print(pred_mask.shape)
    image = image.data.cpu()
    image = image.squeeze()
    pred_mask = torch.argmax(pred_mask,dim=1)
    # pred_mask_1 = torch.argmax(pred_mask_1,dim=1)
    # pred_mask_2 = torch.argmax(pred_mask_2,dim=1)
    # pred_mask_3 = torch.argmax(pred_mask_3,dim=1)


    pred_mask = classes_to_mask(cfg,pred_mask)
    # pred_mask_1 = classes_to_mask(cfg,pred_mask_1)
    # pred_mask_2 = classes_to_mask(cfg,pred_mask_2)
    # pred_mask_3 = classes_to_mask(cfg,pred_mask_3)

    pred_mask = pred_mask.data.cpu()
    # pred_mask_1 = pred_mask_1.data.cpu()
    # pred_mask_2 = pred_mask_2.data.cpu()
    # pred_mask_3 = pred_mask_3.data.cpu()

    pred_mask = pred_mask.squeeze().numpy()
    # pred_mask_1 = pred_mask_1.squeeze().numpy()
    # pred_mask_2 = pred_mask_2.squeeze().numpy()
    # pred_mask_3 = pred_mask_3.squeeze().numpy()


    # (h, w) = image.shape[:2]
    # (cX, cY) = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D((cX, cY), -25, 1.0)
    # pred_mask_1 = cv2.warpAffine(pred_mask_1, M, (w, h))
    # M = cv2.getRotationMatrix2D((cX, cY), 25, 1.0)
    # pred_mask_2 = cv2.warpAffine(pred_mask_2, M, (w, h))
    # pred_mask_3 = cv2.flip(pred_mask_3, 0)

    pred_mask = np.ma.masked_where(pred_mask == 0, pred_mask)
    # pred_mask_1 = np.ma.masked_where(pred_mask_1 == 0, pred_mask_1)
    # pred_mask_2 = np.ma.masked_where(pred_mask_2 == 0, pred_mask_2)
    # pred_mask_3 = np.ma.masked_where(pred_mask_3 == 0, pred_mask_3)
    
    
    fig, ax1 = plt.subplots(1,1)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax1.imshow(image,cmap="gray",interpolation='none')
    ax1.imshow(pred_mask,cmap="jet",interpolation='none', alpha = 0.5)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    # ax2.imshow(image,cmap="gray",interpolation='none')
    # ax2.imshow(pred_mask_1,cmap="jet",interpolation='none', alpha = 0.5)
    # ax2.axes.xaxis.set_visible(False)
    # ax2.axes.yaxis.set_visible(False)

    # ax3.imshow(image,cmap="gray",interpolation='none')
    # ax3.imshow(pred_mask_2,cmap="jet",interpolation='none', alpha = 0.5)
    # ax3.axes.xaxis.set_visible(False)
    # ax3.axes.yaxis.set_visible(False)

    # ax4.imshow(image,cmap="gray",interpolation='none')
    # ax4.imshow(pred_mask_3,cmap="jet",interpolation='none', alpha = 0.5)
    # ax4.axes.xaxis.set_visible(False)
    # ax4.axes.yaxis.set_visible(False)
    plt.show()


#===================================== Test ====================================#
def test(cfg, unet_path,test_loader):
    unet = build_model(cfg)
    if os.path.isfile(unet_path):
        # Load the pretrained Encoder
        unet.load_state_dict(torch.load(unet_path))
        print('%s is Successfully Loaded from %s'%(cfg.model_type,unet_path))
    unet.eval()
    test_len = len(test_loader)
    length = 0
    results = [] 
    for images in tqdm(
			test_loader, 
			total = test_len, 
			desc="Predict Round", 
			unit="batch", 
			leave=False):
        image = images.to(cfg.device)
        # image_1 = images[1].to(cfg.device)
        # image_2 = images[2].to(cfg.device)
        # image_3 = images[3].to(cfg.device)


        length+=1
        with torch.no_grad():

            pred_mask=unet(image)
            # pred_mask_1=unet(image_1)
            # pred_mask_2=unet(image_2)
            # pred_mask_3=unet(image_3)

        save_validation_results(cfg,images, pred_mask,length)#pred_mask_1,pred_mask_2,pred_mask_3,length)

    return pred_mask

# rt_struct = RT_Struct(pred_masks)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

        
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')  
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    # misc
    parser.add_argument('--mode', type=str, default='predict')
    parser.add_argument('--model_name', type=str, default='ResAttU_Net-200-0.0010-15_0.pkl')
    parser.add_argument('--model_type', type=str, default='ResAttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\ResAttU_Net\\24_3_multiclass_200_4')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\multiclass_not_4\\test')
    parser.add_argument('--result_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\predict')

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND","RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")
    parser.add_argument('--encoder_name', type=str, default='resnet152', help="Set an encoder (It works only in UNet, UNet++, DeepLabV3, and DeepLab+V3)")
    parser.add_argument('--encoder_weights', type=str, default=None, help="Pretrained weight, default: Random Init")
    parser.add_argument("--smp", action="store_true", help="Use smp_library")


    config = parser.parse_args()
    config.dropout = 0
    unet_path = os.path.join(
			config.model_path, config.model_name
			)
    test_loader = get_loader(image_path=config.test_path,
                        image_size=config.image_size,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        mode='predict',
                        shuffle=False)

    print(config)
    test(config,unet_path,test_loader)