import argparse
import os
from Multiorgan.multisolver import MultiSolver
from Binary.solver import Solver
from loaders.data_loader import get_loader
from torch.backends import cudnn
import random
from datetime import datetime
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


def main(config):
    print(config.result_path)
    cudnn.benchmark = True
    if config.model_type not in ['U_Net_plus','U_Net','DeepLabV3','ResAttU_Net','DeepLabV3+','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR! Choose the right model')
        return
    
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    

    lr = config.lr 
    print("Learning rate = ", lr)
    epoch = config.num_epochs
    # decay_ratio = random.random()*0.8
    # decay_epoch = int(epoch*decay_ratio)

    print('Defined parameters')
    print(config)    
    classes = class_mapping(config.classes)
    
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            classes = classes,
                            mode='train')
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            classes = classes,
                            mode='valid')

    
    # Train and sample the images
    if config.type == "binary":
        solver = Solver(config, train_loader, valid_loader, None) # Remove test loader from Solver
        if config.mode == 'train':
            solver.train_model()
    elif config.type == "multiclass":
        multisolver = MultiSolver(config, train_loader, valid_loader, classes=classes, save_images=False)
        if config.mode == 'train':
            multisolver.train_model()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--type', type=str, default="multiclass")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')  
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    # misc
    parser.add_argument('--mode', type=str, default='Training')
    parser.add_argument('--model_name', type=str, default='checkpoint.pkl')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net_plus/DeepLabV3/DeepLabV3+/U_Net/R2U_Net/ResAttU_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\U_Net\\9_3_multiclass_200_4')
    parser.add_argument('--train_path', type=str, default='C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\multiclass_not_4\\train')
    parser.add_argument('--valid_path', type=str, default='C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\multiclass_not_4\\validation')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\ek779475\\Desktop\\PRO_pCT_CGFL\\multiclass_not_4\\test')
    parser.add_argument('--result_path', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0., help="Set a dropout value in order to set a dropout layers into the model") 
    parser.add_argument('--encoder_name', type=str, default='resnet152', help="Set an encoder (It works only in UNet, UNet++, DeepLabV3, and DeepLab+V3)")
    parser.add_argument('--encoder_weights', type=str, default=None, help="Pretrained weight, default: Random Init")
    parser.add_argument('--early_stopping', type=int, default=25, help="Set the early stopping `patience` variable")

    # To pass an list argument, you should type
    # i.e. python main.py --classes RECTUM VESSIE TETE_FEMORALE_D TETE_FEMORALE_G
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND", "RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--smp", action="store_true", help="Use smp_library")


    config = parser.parse_args()
    day, month = datetime.date(datetime.now()).day, datetime.date(datetime.now()).month
    config.log_dir = f"./runs/{config.type}/{day}_{month}_{config.model_type}_{config.num_epochs}"
    
    
    if config.smp:
        config.log_dir = f"./runs/{config.type}/{config.encoder_name}_{config.encoder_weights}_{day}_{month}_{config.model_type}_{config.num_epochs}_{config.batch_size}"
        config.result_path=f'./result/{config.model_type}/{config.encoder_name}_{config.encoder_weights}_{day}_{month}_{config.type}_{config.num_epochs}_{config.batch_size}'
    else:
        config.log_dir = f"./runs/{config.type}/{day}_{month}_{config.model_type}_{config.num_epochs}_{config.batch_size}_focal"
        config.result_path=f'./result/{config.model_type}/{day}_{month}_{config.type}_{config.num_epochs}_{config.batch_size}_focal'
    try:
        main(config)
    except KeyboardInterrupt:
        print("Keyboard Interruption")
        exit()