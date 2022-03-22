import os
import numpy as np
import torch
# Fix memory problem
torch.cuda.empty_cache()
import torchvision
from torch import optim
from utils_metrics import DiceBCELoss, collect_metrics
from networks.network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from loaders.data_loader import get_loader
from utils_metrics import AverageMeter

def _update_metricRecords(writer,wr_valid, mode,metric, batch):	

		# writer.add_scalars("loss", {mode:metric.avg_loss}, batch)
		writer.add_scalars("recall", {mode:metric.recall}, batch)
		writer.add_scalars("sensitivity", {mode:metric.sensitivity}, batch)
		writer.add_scalars("specificity", {mode:metric.specificity}, batch)
		writer.add_scalars("dice", {mode:metric.dice}, batch)
		writer.add_scalars("jaccard", {mode:metric.iou}, batch)
		writer.add_scalars("hausdorff", {mode:metric.hausdorff}, batch)
		writer.add_scalars("hausforff_95", {mode:metric.hd95}, batch)
		wr_valid.writerow(
			[
			    batch, 
				metric.precision, metric.recall, metric.sensitivity, 
				metric.specificity, metric.dice, metric.iou,
				metric.hausdorff,metric.hd95
			])
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
    writer = SummaryWriter()
    wr_test = csv.writer(testing_log)
    wr_test.writerow(["batch", "precision", "recall", "sensitivity", "specificity", "dice", "iou","hausdorff_distance","hausdorff_distance_95"])

    # unet_path = os.path.join(unet_path, cfg.model_name)
    # del cfg.unet
    print(unet_path)
    unet = build_model(cfg,device)
    metrics = AverageMeter()
    if os.path.isfile(unet_path):
        # Load the pretrained Encoder
        unet.load_state_dict(torch.load(unet_path))
        print('%s is Successfully Loaded from %s'%(cfg.model_type,unet_path))
    unet.eval()
    test_len = len(test_loader)
    length = 0
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
        length += int(images.size(0)/cfg.batch_size)
        # print(length)
        metrics.update(0, true_masks, pred_masks, images.size(0)/cfg.batch_size)
        _update_metricRecords(writer,wr_test, "Test",metrics, length)
        save_validation_results(cfg,images, pred_masks, true_masks,length)
        
    wr_test.writerow([
        "avg_precision", "avg_recall", "avg_sensitivity", 
        "avg_specificity", "avg_dice", "avg_iou",
        "avg_hd","avg_hd95"
        ])
    wr_test.writerow([ 
        metrics.avg_precision, metrics.avg_recall, metrics.avg_sensitivity, 
        metrics.avg_specificity, metrics.avg_dice, metrics.avg_iou,
        metrics.avg_hausdorff,metrics.avg_hd95
        ])

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
    parser.add_argument('--model_name', type=str, default='U_Net-80-0.0010-13_4_batch.pkl')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='.\\models')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\RECTUM\\test')
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