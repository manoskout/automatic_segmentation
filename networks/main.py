import argparse
import os
from re import I

from sklearn.model_selection import KFold
import torch
from Multiorgan.multisolver import MultiSolver
# from Binary.solver import Solver
from loaders.data_loader import ImageFolder, ImageFolder2_5D
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
    
    

    
    # Train and sample the images

        
    if config.mode == 'train':
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=config.k_folds, shuffle=False)
        if config.strategy == "2D":
            dataset = ImageFolder(
                root = config.train_path, 
                image_size =config.image_size, 
                mode=config.mode,
                augmentation_prob=0.4, 
                is_multiorgan=True, 
                classes=classes
                )
        elif config.strategy == "2_5D":
            dataset = ImageFolder2_5D(
                root = config.train_path, 
                image_size =config.image_size, 
                mode=config.mode,
                augmentation_prob=0.4, 
                is_multiorgan=True, 
                classes=classes
                )
        if config.type == "multiclass":
            classes = classes
            del classes[0] # Delete the background label
        else:
            classes = []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            valid_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle = False,
                sampler = train_subsampler)
            valid_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle = False,
                sampler = valid_subsampler)
            # Solver(config, train_loader, valid_loader, None) if config.type == "binary" else 
            solver = MultiSolver(config, train_loader, valid_loader, classes, False)
            print("K-Fold cross validation. Current fold: {}".format(fold))
            solver.train_model(fold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--type', type=str, default="multiclass")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')  
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=200)
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
<<<<<<< HEAD
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--train_path', type=str, default='/home/mkout/PRO_pCT_CGFL/2_5D_multiclass_imbalanced/train')
    parser.add_argument('--valid_path', type=str, default='/home/mkout/PRO_pCT_CGFL/2_5D_multiclass_imbalanced/validation')
    parser.add_argument('--test_path', type=str, default='/home/mkout/PRO_pCT_CGFL/2_5D_multiclass_imbalanced/test')
    parser.add_argument('--result_path', type=str, default='')
=======
    parser.add_argument('--model_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\networks\\result\\U_Net\\9_3_multiclass_200_4')
    parser.add_argument('--train_path', type=str, default='/home/mkout/PRO_pCT_CGFL/2_5D_multiclass_imbalanced/train')
    parser.add_argument('--valid_path', type=str, default='/home/mkout/PRO_pCT_CGFL/2_5D_multiclass_imbalanced/validation')
    parser.add_argument('--test_path', type=str, default='/home/mkout/PRO_pCT_CGFL/2_5D_multiclass_imbalanced/test')
    parser.add_argument('--result_path', type=str, default='/home/mkout/automatic_segmentation/networks/results')
>>>>>>> b2e04e269346fab7c7df384c1630ca4287b1114c
    parser.add_argument('--dropout', type=float, default=0., help="Set a dropout value in order to set a dropout layers into the model")
    parser.add_argument('--norm', type=str, default='batch', help="Set an normalization method (Default: BatchNormalization),\n\t Available choices: \n\t batch: BatchNormalization, group: GroupNormalization, instance: InstanceNormalization")
     
    parser.add_argument('--encoder_name', type=str, default='resnet152', help="Set an encoder (It works only in UNet, UNet++, DeepLabV3, and DeepLab+V3)")
    parser.add_argument('--encoder_weights', type=str, default=None, help="Pretrained weight, default: Random Init")
    parser.add_argument('--early_stopping', type=int, default=25, help="Set the early stopping `patience` variable")
    parser.add_argument('--k_folds', type=int, default=5, help="Set the early stopping `patience` variable")

    # To pass an list argument, you should type
    # i.e. python main.py --classes RECTUM VESSIE TETE_FEMORALE_D TETE_FEMORALE_G
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND", "RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--smp", action="store_true", help="Use smp_library")
    parser.add_argument("--strategy", type=str, default="2_5D", help="Training strategy (default: 2_5D), choices 2.5D, 2D")



    config = parser.parse_args()
    day, month = datetime.date(datetime.now()).day, datetime.date(datetime.now()).month
    config.log_dir = f"./runs/{config.type}/{day}_{month}_{config.model_type}_{config.num_epochs}"
    
    
    if config.smp:
        config.log_dir = f"./runs/{config.type}/{config.encoder_name}_{config.encoder_weights}_{day}_{month}_{config.model_type}_{config.num_epochs}_{config.batch_size}_{config.strategy}"
        config.result_path=f'./result/{config.model_type}/{config.encoder_name}_{config.encoder_weights}_{day}_{month}_{config.type}_{config.num_epochs}_{config.batch_size}_{config.strategy}'
    else:
        config.log_dir = f"./runs/{config.type}/{day}_{month}_{config.model_type}_{config.num_epochs}_{config.batch_size}_{config.norm}_{config.strategy}"
        config.result_path=f'./result/{config.model_type}/{day}_{month}_{config.type}_{config.num_epochs}_{config.batch_size}_{config.norm}_{config.strategy}'
    try:
        main(config)
    except KeyboardInterrupt:
        print("Keyboard Interruption")
        exit()