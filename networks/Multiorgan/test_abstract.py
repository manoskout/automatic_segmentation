import os
import numpy as np
import torch
import sys
# Magic - Finding the parent directory automatically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_metrics import AverageMeter
# Fix memory problem
torch.cuda.empty_cache()
from networks.network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from loaders.data_loader import get_loader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from utils import classes_to_mask, class_mapping, build_model

import pandas as pd
def save_validation_results(cfg,image,true_mask,pred25d_mask,counter, metric, classes):#pred_mask_1,pred_mask_2,pred_mask_3,counter = 0 ):

    image = image.data.cpu()
    image = image.squeeze()
    true_mask = true_mask.data.cpu()
    true_mask = true_mask.squeeze()
    true_mask = classes_to_mask(cfg,true_mask)

    # pred2d_mask = torch.argmax(pred2d_mask,dim=1)
    # pred2d_mask = classes_to_mask(cfg,pred2d_mask)

    # pred2d_mask = pred2d_mask.data.cpu()
    # pred2d_mask = pred2d_mask.squeeze().numpy()
    # pred2d_mask = np.ma.masked_where(pred2d_mask == 0, pred2d_mask)
    
    pred25d_mask = torch.argmax(pred25d_mask,dim=1)
    pred25d_mask = classes_to_mask(cfg,pred25d_mask)

    pred25d_mask = pred25d_mask.data.cpu()
    pred25d_mask = pred25d_mask.squeeze().numpy()
    pred25d_mask = np.ma.masked_where(pred25d_mask == 0, pred25d_mask)

    true_mask = np.ma.masked_where(true_mask == 0, true_mask)
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    


    plt.subplot(1,2,1)
    plt.title("Ground Truth")
    plt.imshow(image,cmap="gray",interpolation='none')
    plt.imshow(true_mask,cmap="cool",interpolation='none', alpha = 0.5)
    plt.axis('off')

    # plt.subplot(1,3,2)
    # plt.title("2D Prediction")
    # plt.imshow(image,cmap="gray",interpolation='none')
    # plt.imshow(pred2d_mask,cmap="cool",interpolation='none', alpha = 0.5)
    # plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("2.5D Prediction")
    plt.imshow(image,cmap="gray",interpolation='none')
    plt.imshow(pred25d_mask,cmap="cool",interpolation='none', alpha = 0.5)
    plt.axis('off')
    plt.show()

#===================================== Test ====================================#
def test_abs(config2D,config25D):
    unet2D_path = os.path.join(config2D.model_path, config2D.model_name)
    unet25D_path = os.path.join(config25D.model_path, config25D.model_name)  
    test2D_loader = get_loader(image_path=config2D.test_path,
                        image_size=config2D.image_size,
                        batch_size=config2D.batch_size,
                        num_workers=config2D.num_workers,
                        classes = config2D.classes,
                        mode='test',
                        strategy=config2D.strategy,
                        shuffle=False)
    test25D_loader = get_loader(image_path=config25D.test_path,
                        image_size=config25D.image_size,
                        batch_size=config25D.batch_size,
                        num_workers=config25D.num_workers,
                        classes = config25D.classes,
                        mode='test',
                        strategy=config25D.strategy,
                        shuffle=False)

    
    # metrics = AverageMeter()
    raunet2D = build_model(config2D)
    raunet25D = build_model(config25D)

    if os.path.isfile(unet2D_path):
        # Load the pretrained Encoder
        if config2D.device == "cuda":
            raunet2D.load_state_dict(torch.load(unet2D_path))
            raunet25D.load_state_dict(torch.load(unet25D_path))

        else:
            raunet2D.load_state_dict(torch.load(unet2D_path,map_location=torch.device('cpu')))
            raunet25D.load_state_dict(torch.load(unet25D_path,map_location=torch.device('cpu')))

        print('%s is Successfully Loaded from %s'%(config2D.model_type,unet2D_path))
        print('%s is Successfully Loaded from %s'%(config25D.model_type,unet25D_path))

    raunet2D.eval()
    raunet25D.eval()

    test_len = len(test2D_loader)
    length = 0

    for ((image2D, true_mask2D),(image25D, true_mask25D)) in tqdm(
			zip(test2D_loader,test25D_loader), 
			total = test_len, 
			desc="Test Round", 
			unit="batch", 
			leave=False):
        image2D = image2D.to(config2D.device)
        true_mask2D = true_mask2D.to(config2D.device)
        image25D = image25D.to(config25D.device)
        true_mask25D = true_mask25D.to(config25D.device)

        with torch.no_grad():
            pred2D_mask = raunet2D(image2D)
            pred25D_mask = raunet25D(image25D)

        
        # metrics.update(0, true_mask, pred_mask, image.size(0), classes=cfg.classes) 

        save_validation_results(config25D,image25D[:,1,:,:],true_mask25D, pred2D_mask,pred25D_mask,length,None, config25D.classes)#pred_mask_1,pred_mask_2,pred_mask_3,length)

        length += image2D.size(0)/config2D.batch_size
        # metrics.reset()

#===================================== Test ====================================#
def test(config25D):
    unet25D_path = os.path.join(config25D.model_path, config25D.model_name)  

    test25D_loader = get_loader(image_path=config25D.test_path,
                        image_size=config25D.image_size,
                        batch_size=config25D.batch_size,
                        num_workers=config25D.num_workers,
                        classes = config25D.classes,
                        mode='test',
                        strategy=config25D.strategy,
                        shuffle=False)

    
    # metrics = AverageMeter()
    raunet25D = build_model(config25D)

    if os.path.isfile(unet25D_path):
        # Load the pretrained Encoder
        if config25D.device == "cuda":
            raunet25D.load_state_dict(torch.load(unet25D_path))

        else:
            raunet25D.load_state_dict(torch.load(unet25D_path,map_location=torch.device('cpu')))


        print('%s is Successfully Loaded from %s'%(config25D.model_type,unet25D_path))
    raunet25D.eval()

    test_len = len(test25D_loader)
    length = 0

    for (image25D, true_mask25D) in tqdm(
		    test25D_loader, 
			total = test_len, 
			desc="Test Round", 
			unit="batch", 
			leave=False):
        image25D = image25D.to(config25D.device)
        true_mask25D = true_mask25D.to(config25D.device)

        with torch.no_grad():
            pred25D_mask = raunet25D(image25D)

        

        save_validation_results(config25D,image25D[:,1,:,:],true_mask25D,pred25D_mask,length,None, config25D.classes)#pred_mask_1,pred_mask_2,pred_mask_3,length)

        length += image25D.size(0)/config25D.batch_size
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
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='checkpoint.pkl')
    parser.add_argument('--model_type', type=str, default='ResAttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='/Users/manoskoutoulakis/Desktop/test_set/diceLoss_ResAttUNet_2D')
    parser.add_argument('--test_path', type=str, default='/Users/manoskoutoulakis/Desktop/test_set/dt/2D')
    parser.add_argument('--result_path', type=str, default="/Users/manoskoutoulakis/Desktop/test_set/abs_res")

    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND","RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")
    parser.add_argument('--encoder_name', type=str, default='resnet152', help="Set an encoder (It works only in UNet, UNet++, DeepLabV3, and DeepLab+V3)")
    parser.add_argument('--encoder_weights', type=str, default=None, help="Pretrained weight, default: Random Init")
    parser.add_argument("--smp", action="store_true", help="Use smp_library")
    parser.add_argument("--strategy", type=str, default="2D", help="Training strategy (default: 2.5D), choices 2.5D, 2D")
    config = parser.parse_args()
    config.dropout = 0.

    config.classes = class_mapping(config.classes)
    config.norm = "batch"
    del config.classes[0] # Delete the background

    # Deep copy of argument from argparser 
    config2D= argparse.Namespace(**vars(config))
    config25D = argparse.Namespace(**vars(config))
    config25D.norm = "batch"

    config25D.model_path = '/Users/manoskoutoulakis/Desktop/test_set/focalTversky_ResAttUnet_2_5D'
    config25D.test_path = "/Users/manoskoutoulakis/Desktop/test_set/dt/25D"
    config25D.strategy = "2_5D"
    config25D.img_ch = 3

    print(f"2d strategy: {config2D.strategy}")
    print(f"25d strategy: {config25D.strategy}")

    try:
        # test_abs(config2D,config25D)
        test(config25D)

    except KeyboardInterrupt:
        os._exit(1)