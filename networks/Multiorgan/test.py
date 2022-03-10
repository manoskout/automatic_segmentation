import os
import numpy as np
import torch
import sys
# Magic - Finding the parent directory automatically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_metrics import AverageMeter
# Fix memory problem
torch.cuda.empty_cache()
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from loaders.data_loader import get_loader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


def class_mapping(classes):
	""" Maps the classes according to the pixel are shown into the mask
	"""
	mapping_dict={}
	for index,i in enumerate(classes):
		
		if index == 0:
			mapping_dict[0]= index
		else:
			mapping_dict[int(255/index)]= index
	return mapping_dict

def build_model(cfg, device):
    """Build generator and discriminator."""
    if cfg.smp:
        if cfg.model_type == "DeepLabV3+":
            unet = smp.DeepLabV3Plus(
                encoder_name=cfg.encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=cfg.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=cfg.img_ch,        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=cfg.output_ch,         # model output channels (number of classes in your dataset)
            )
        elif cfg.model_type == "U_Net":
            unet = smp.Unet(
                encoder_name=cfg.encoder_name,       
                encoder_weights=cfg.encoder_weights,    
                in_channels=cfg.img_ch,       
                classes=cfg.output_ch,         
            )
        elif cfg.model_type == "U_Net_plus":
            unet = smp.UnetPlusPlus(
                encoder_name=cfg.encoder_name,        
                encoder_weights=cfg.encoder_weights,    
                in_channels=cfg.img_ch,       
                classes=cfg.output_ch,         
            )
    elif cfg.model_type =='U_Net':
        unet = U_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch)
    elif cfg.model_type =='R2U_Net':
        unet = R2U_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch,t=cfg.t)
    elif cfg.model_type =='AttU_Net':
        unet = AttU_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch)
    elif cfg.model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch,t=cfg.t)
        
    unet.to(device)

    return unet
def classes_to_mask(cfg: argparse.Namespace, mask : torch.Tensor) -> torch.Tensor:
    """Converts the labeled pixels to range 0-255
    """
    for index, k in enumerate(cfg.classes):
        # prin0t(index,"ind : ",int(255/index)) if index != 0 else print(index,"ind : ","0")
        mask[mask==index] = int(255/index) if index != 0 else 0


    return mask.type(torch.float)
def save_validation_results(cfg,image, pred_mask, true_mask,epoch):
    if len(cfg.classes)>1:
        pred_mask = torch.argmax(pred_mask,dim=1)
        pred_mask = classes_to_mask(cfg,pred_mask)
        true_mask[0] = classes_to_mask(cfg, true_mask[0])
        true_mask = true_mask.to(torch.float32)
    image = image.data.cpu()
    pred_mask = pred_mask.data.cpu()
    true_mask = true_mask.data.cpu()
    fig, ax = plt.subplots(1,1)
    pred_mask = pred_mask.squeeze()
    true_mask = true_mask.squeeze()
    pred_mask = np.ma.masked_where(pred_mask == 0, pred_mask)
    true_mask = np.ma.masked_where(true_mask == 0, true_mask)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax.imshow(image.squeeze(),cmap="gray",interpolation='none')
    ax.imshow(true_mask,cmap="hot",interpolation='none', alpha = 0.8)
    ax.imshow(pred_mask,cmap="jet",interpolation='none', alpha = 0.5)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(os.path.join(
            cfg.result_path,
            '%s_test_%d_result_INPUT.png'%(cfg.model_type,epoch+1)
            )
    )
    # plt.grid(False)
    # plt.show()
    

def _update_metricRecords(tens_writer,csv_writer,mode,metric, classes=None, step=None) -> None:	
    avg_metrics = [
                metric.precision.mean(), metric.recall.mean(), metric.sensitivity.mean(),
                metric.specificity.mean(), metric.dice.mean(), metric.iou.mean(),
                metric.hd.mean(),metric.hd95.mean()
            ]
    
    if classes:
        for index in range(len(classes.items())):
            avg_metrics.append(metric.iou[index])
            avg_metrics.append(metric.dice[index])
            avg_metrics.append(metric.hd[index])

    #     tens_writer.add_scalars("recall", {mode:metric.all_recall}, step)
    #     tens_writer.add_scalars("sensitivity", {mode:metric.all_sensitivity}, step)
    #     tens_writer.add_scalars("specificity", {mode:metric.all_specificity}, step)
    #     tens_writer.add_scalars("dice", {mode:metric.all_dice}, step)
    #     tens_writer.add_scalars("jaccard", {mode:metric.all_iou}, step)
    #     tens_writer.add_scalars("hausdorff", {mode:metric.all_hd}, step)
    #     tens_writer.add_scalars("hausforff_95", {mode:metric.all_hd95}, step)
        
    #     if classes:
    #         for _,index in classes.items():
    #             # print(index,"rrrrrrrrrrrrrr")
    #             index = int(index)
    #             tens_writer.add_scalars(f"iou_{index}",metric.avg_iou[index], step)
    #             tens_writer.add_scalars(f"dice_{index}",metric.avg_dice[index], step)
    #             tens_writer.add_scalars(f"hd_{index}",metric.avg_hd[index], step)
    csv_writer.writerow( 
            avg_metrics
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
    metric_list = ["precision", "recall", "sensitivity", "specificity", "dice", "iou","hd","hd95"]
    if cfg.classes:
        # print(cfg.classes)
        for _,id in cfg.classes.items():
            for i in ["iou","dice","hd"]:
                metric_list.append(f"{id}_{i}")
    wr_test.writerow(metric_list)

    writer = SummaryWriter()
    metrics = AverageMeter()
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
    for (image, true_mask) in tqdm(
			test_loader, 
			total = test_len, 
			desc="Test Round", 
			unit="batch", 
			leave=False):
        image = image.to(device)
        true_mask = true_mask.to(device)
        with torch.no_grad():
            pred_mask = unet(image)
        metrics.update(0, true_mask, pred_mask, image.size(0), classes=cfg.classes)        
        length += image.size(0)/cfg.batch_size
        _update_metricRecords(writer,wr_test,"Testing",metrics, classes=cfg.classes, step=test_len-length)
        save_validation_results(cfg,image, pred_mask, true_mask,test_len-length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')  
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='U_Net_plus-200-0.0010-5.pkl')
    parser.add_argument('--model_type', type=str, default='U_Net_plus', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\U_Net_plus\\resnet152_None_10_3_multiclass_200_4')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\test')
    parser.add_argument('--result_path', type=str, default='')

    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND","RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")
    parser.add_argument('--encoder_name', type=str, default='resnet152', help="Set an encoder (It works only in UNet, UNet++, DeepLabV3, and DeepLab+V3)")
    parser.add_argument('--encoder_weights', type=str, default=None, help="Pretrained weight, default: Random Init")
    parser.add_argument("--smp", action="store_true", help="Use smp_library")

    config = parser.parse_args()
    config.result_path = config.model_path
    unet_path = os.path.join(config.model_path, config.model_name)
    config.classes = class_mapping(config.classes)
    test_loader = get_loader(image_path=config.test_path,
                        image_size=config.image_size,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        classes = config.classes,
                        mode='test')
                        
    device = "cuda"
    test(config,unet_path,test_loader, config.result_path, device)