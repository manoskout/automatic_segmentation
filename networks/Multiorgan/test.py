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

def save_validation_results(cfg,image,true_mask, pred_mask,counter=0):#pred_mask_1,pred_mask_2,pred_mask_3,counter = 0 ):

    image = image.data.cpu()
    image = image.squeeze()
    true_mask = true_mask.data.cpu()
    true_mask = true_mask.squeeze()

    pred_mask = torch.argmax(pred_mask,dim=1)
    pred_mask = classes_to_mask(cfg,pred_mask)
    true_mask = classes_to_mask(cfg,true_mask)

    pred_mask = pred_mask.data.cpu()
    pred_mask = pred_mask.squeeze().numpy()
    pred_mask = np.ma.masked_where(pred_mask == 0, pred_mask)
    true_mask = np.ma.masked_where(true_mask == 0, true_mask)
    
    fig, (ax1,ax2) = plt.subplots(1,2)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax1.imshow(image,cmap="gray",interpolation='none')
    ax1.imshow(true_mask,cmap="cool",interpolation='none', alpha = 0.5)
    
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)

    ax2.imshow(image,cmap="gray",interpolation='none')
    ax2.imshow(pred_mask,cmap="cool",interpolation='none', alpha = 0.5)

    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    

    plt.show()

def _update_metricRecords(writer, csv_writer, metric, mode="test", classes=None, img_num=1) -> None:	
    # avg_metrics = [
    #             metric.all_precision, metric.all_recall, metric.all_sensitivity, 
    #             metric.all_specificity, metric.all_dice, metric.all_iou,
    #             metric.all_hd,metric.all_hd95
    #         ]
    avg_metrics = [metric.all_dice, metric.all_iou, metric.all_hd95]
    if classes:
        for index in range(len(classes.items())):
            try:
                avg_metrics.append(metric.iou[index])
                avg_metrics.append(metric.dice[index])
                avg_metrics.append(metric.hd95[index])
            except IndexError:
                print("IndexError")
                avg_metrics.append(np.nan)
                avg_metrics.append(np.nan)
                avg_metrics.append(np.nan)
        # writer.add_scalars("recall", {mode:metric.all_recall}, img_num)
        # writer.add_scalars("sensitivity", {mode:metric.all_sensitivity}, img_num)
        # writer.add_scalars("specificity", {mode:metric.all_specificity}, img_num)
        # writer.add_scalars("dice", {mode:metric.all_dice}, img_num)
        # writer.add_scalars("jaccard", {mode:metric.all_iou}, img_num)
        # writer.add_scalars("hausdorff", {mode:metric.all_hd}, img_num)
        # writer.add_scalars("hausforff_95", {mode:metric.all_hd95}, img_num)
    csv_writer.writerow( 
        avg_metrics
        )
    # print(f"Testing Res: dice: {metric.all_dice}, iou: {metric.all_iou}, hd: {metric.all_hd} ")

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
    print(f"Metrics collector path: {testing_log}")
    wr_test = csv.writer(testing_log)
    metric_list = ["dice","iou","hd95"] #["precision", "recall", "sensitivity", "specificity", "dice", "iou","hd","hd95"]
    # metric_list = []
    if cfg.classes:
        # print(cfg.classes)
        for _,id in cfg.classes.items():
            for i in ["iou","dice","hd95"]:
                metric_list.append(f"{id}_{i}")
    wr_test.writerow(metric_list)

    writer = SummaryWriter()
    metrics = AverageMeter()
    # unet_path = os.path.join(unet_path, cfg.model_name)
    # del cfg.unet
    print(unet_path)
    unet = build_model(cfg)
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
        # print(f"\niou: {metrics.iou}, \ndice: {metrics.dice}, \nHD: {metrics.hd95}")

        save_validation_results(cfg,image,true_mask, pred_mask,length)#pred_mask_1,pred_mask_2,pred_mask_3,length)

        length += image.size(0)/cfg.batch_size
        _update_metricRecords(writer,wr_test,metrics, classes=cfg.classes, img_num=test_len-length)
        metrics.reset()

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
    parser.add_argument('--model_name', type=str, default='ResAttU_Net-200-0.0010-15_0.pkl')
    parser.add_argument('--model_type', type=str, default='ResAttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\U_Net\\24_3_multiclass_200_4')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\multiclass_imbalanced\\test')
    parser.add_argument('--result_path', type=str, default='C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\multiclass_imbalanced\\metrics')

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND","RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")
    parser.add_argument('--encoder_name', type=str, default='resnet152', help="Set an encoder (It works only in UNet, UNet++, DeepLabV3, and DeepLab+V3)")
    parser.add_argument('--encoder_weights', type=str, default=None, help="Pretrained weight, default: Random Init")
    parser.add_argument("--smp", action="store_true", help="Use smp_library")

    config = parser.parse_args()
    config.dropout = 0.
    # config.result_path = config.model_path
    unet_path = os.path.join(config.model_path, config.model_name)
    config.classes = class_mapping(config.classes)
    del config.classes[0] # Delete the background
    test_loader = get_loader(image_path=config.test_path,
                        image_size=config.image_size,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        classes = config.classes,
                        mode='test')
                        
    device = "cuda"
    test(config,unet_path,test_loader, config.result_path, device)