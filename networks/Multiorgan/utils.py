import segmentation_models_pytorch as smp 
from networks.network import U_Net,R2U_Net,AttU_Net,R2AttU_Net, ResAttU_Net 
from torch import optim
import argparse
import torch
def build_model(cfg: argparse.Namespace):
    """Build generator and discriminator."""
    unet = None
    #  TODO -> Dropout layers are not implemented into the R2U_net and R2Att_unet
    if cfg.smp:
        if cfg.model_type == "DeepLabV3+":
            unet = smp.DeepLabV3Plus(
                encoder_name=cfg.encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights= cfg.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=cfg.img_ch,        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=cfg.output_ch,         # model output channels (number of classes in your dataset)
            )
        elif cfg.model_type == "U_Net_plus":
            unet = smp.UnetPlusPlus(
                encoder_name=cfg.encoder_name,        
                encoder_weights=cfg.encoder_weights,    
                in_channels=cfg.img_ch,       
                classes=cfg.output_ch,         
            )
    elif cfg.model_type =='U_Net':
        unet = U_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch, dropout=cfg.dropout)
    elif cfg.model_type =='R2U_Net':
        unet = R2U_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch,t=cfg.t)
    elif cfg.model_type =='AttU_Net':
        unet = AttU_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch, dropout=cfg.dropout)
    elif cfg.model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch,t=cfg.t)
    elif cfg.model_type == 'ResAttU_Net':
        unet = ResAttU_Net(img_ch=cfg.img_ch,output_ch=cfg.output_ch,t=cfg.t)
        
    # cfg.grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    return unet.to(cfg.device)

def class_mapping(classes:list)-> dict:
	""" Maps the classes according to the pixel are shown into the mask
	"""
	mapping_dict={}
	for index,i in enumerate(classes):
		
		if index == 0:
			mapping_dict[0]= index
		else:
			mapping_dict[int(255/index)]= index
	return mapping_dict

def classes_to_mask(cfg: argparse.Namespace, mask : torch.Tensor) -> torch.Tensor:
    """Converts the labeled pixels to range 0-255
    """
    for index, k in enumerate(cfg.classes):
        # prin0t(index,"ind : ",int(255/index)) if index != 0 else print(index,"ind : ","0")
        mask[mask==index] = int(255/index) if index != 0 else 0
    return mask.type(torch.float)
