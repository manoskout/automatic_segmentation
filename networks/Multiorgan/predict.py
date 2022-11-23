import os
from re import L
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_metrics import AverageMeter
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
from PIL import Image

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
def save_validation_results(cfg,image, pred_mask,counter=0, res_path=""):#pred_mask_1,pred_mask_2,pred_mask_3,counter = 0 ):

    image = image.data.cpu()
    image = image.squeeze()
    pred_mask = torch.argmax(pred_mask,dim=1)
    pred_mask = classes_to_mask(cfg,pred_mask)
    pred_mask = pred_mask.data.cpu()
    pred_mask = pred_mask.squeeze().numpy()
    print(np.unique(pred_mask))
    # pred_mask = np.ma.masked_where(pred_mask == 0, pred_mask)
    # im = Image.fromarray(pred_mask)
    # result_f = os.path.join(res_path, f"mask{counter}.tiff")
    # im.save(result_f)
    fig, ax1 = plt.subplots(1,1)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax1.imshow(image,cmap="gray",interpolation='none')
    ax1.imshow(pred_mask,cmap="jet",interpolation='none', alpha = 0.5)
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    plt.show()


#===================================== Test ====================================#
def test(cfg, unet_path,test_loader):
    unet = build_model(cfg)
    if os.path.isfile(unet_path):
        # Load the pretrained Encoder
        if cfg.device =="cuda":
            unet.load_state_dict(torch.load(unet_path))
        else:
            unet.load_state_dict(torch.load(unet_path,map_location=torch.device('cpu')))
        print('%s is Successfully Loaded from %s'%(cfg.model_type,unet_path))
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

        length+=1
        unet.eval()

        with torch.no_grad():
            pred_mask=unet(image)

        save_validation_results(cfg,images, pred_mask,length, cfg.result_path)#pred_mask_1,pred_mask_2,pred_mask_3,length)

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
    parser.add_argument('--model_name', type=str, default='checkpoint.pkl')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='/Users/manoskoutoulakis/Desktop/presentation_set/for_presentation')
    parser.add_argument('--test_path', type=str, default='/Users/manoskoutoulakis/Desktop/test')
    parser.add_argument('--result_path', type=str, default='/Users/manoskoutoulakis/Desktop/test')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND","RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")
    parser.add_argument('--encoder_name', type=str, default='resnet152', help="Set an encoder (It works only in UNet, UNet++, DeepLabV3, and DeepLab+V3)")
    parser.add_argument('--encoder_weights', type=str, default=None, help="Pretrained weight, default: Random Init")
    parser.add_argument("--smp", action="store_true", help="Use smp_library")


    config = parser.parse_args()
    config.dropout = 0
    unet_path = os.path.join(
			config.model_path, config.model_name
			)
    del config.classes[0] # Delete the background

    test_loader = get_loader(image_path=config.test_path,
                        image_size=config.image_size,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        mode='predict',
                        shuffle=False)

    print(config)
    test(config,unet_path,test_loader)