import os
import numpy as np
import torch
# Fix memory problem
torch.cuda.empty_cache()
import torchvision
from torch import optim
from utils_metrics import DiceBCELoss, collect_metrics
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from loaders.data_loader import get_loader


def build_model(cfg, device):
    """Build generator and discriminator."""
    if cfg.model_type =='U_Net':
        unet = U_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch)
    elif cfg.model_type =='R2U_Net':
        unet = R2U_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch,t=cfg.t)
    elif cfg.model_type =='AttU_Net':
        unet = AttU_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch)
    elif cfg.model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch,t=cfg.t)
        
    unet.to(device)

    return unet

def save_validation_results(cfg,image, pred_mask, true_mask,epoch):
    image = image.data.cpu()
    pred_mask = pred_mask.data.cpu()
    true_mask = true_mask.data.cpu()
    torchvision.utils.save_image(
        image,
        os.path.join(
            cfg.result_path,
            '%s_test_%d_result_INPUT.png'%(cfg.model_type,epoch+1
            )
        )
    )
    torchvision.utils.save_image(
        pred_mask,
        os.path.join(
            cfg.result_path,
            '%s_test_%d_result_PRED.png'%(cfg.model_type,epoch+1
            )
        )
    )
    torchvision.utils.save_image(
        true_mask,
        os.path.join(
            cfg.result_path,
            '%s_test_%d_result_GT.png'%(cfg.model_type,epoch+1
            )
        )
    )
#===================================== Test ====================================#
def test(cfg, unet_path,test_loader, test_save_path, device ="cuda"):
    testing_log = open(
        os.path.join(
            test_save_path,
            'result_testing.csv'
        ), 
        'a', 
        encoding='utf-8', 
        newline=''
    )
    wr_test = csv.writer(testing_log)

    # unet_path = os.path.join(unet_path, cfg.model_name)
    # del cfg.unet
    print(unet_path)
    unet = build_model(cfg,device)
    if os.path.isfile(unet_path):
        # Load the pretrained Encoder
        unet.load_state_dict(torch.load(unet_path))
        print('%s is Successfully Loaded from %s'%(cfg.model_type,unet_path))
    unet.eval()
    test_len = len(test_loader)
    length = 0
    dice_c = iou = 0.	
    for (images, true_masks) in tqdm(
			test_loader, 
			total = test_len, 
			desc="Test Round", 
			unit="batch", 
			leave=False):
        images = images.to(device)
        true_masks = true_masks.to(device)
        with torch.no_grad():
            pred_masks = unet(images)
        _, _, _, _, dice_c, iou =  collect_metrics(true_masks,pred_masks)
        
        wr_test.writerow([dice_c, iou])
        length += images.size(0)/cfg.batch_size
        save_validation_results(cfg,images, pred_masks, true_masks,test_len-length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')  
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='U_Net-80-0.0004-52.pkl')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='.\\models')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\ek779475\\Desktop\\RECTUM\\automatic_segmentation\\Dataset\\RECTUM\\test')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    unet_path = os.path.join(
			config.model_path, config.model_name
			)
    test_loader = get_loader(image_path=config.test_path,
                        image_size=config.image_size,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        mode='test')
                        
    device = "cuda"
    test(config,unet_path,test_loader, config.result_path, device)