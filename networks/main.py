import argparse
import os
from Multiorgan.multisolver import MultiSolver
from Binary.solver import Solver
from loaders.data_loader import get_loader
from torch.backends import cudnn
import random
def class_mapping(classes):
	""" Maps the classes according to the pixel are shown into the mask
	"""
	mapping_dict={}
	for index,i in enumerate(classes):
		
		if index == 0:
			mapping_dict[0]= index
		else:
			mapping_dict[int(255/index)]= index
		# print(f"{index} : {i} --> {mapping_dict}")
	return mapping_dict


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR! Choose the right model')
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    
    lr = config.lr 
    print("Learning rate = ", lr)
    epoch = config.num_epochs
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.num_epochs_decay = decay_epoch

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
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            classes = classes,
                            mode='test')

 

    
    # Train and sample the images
    if config.type == "binary":
        solver = Solver(config, train_loader, valid_loader, test_loader)
        if config.mode == 'train':
            solver.train_model()
        elif config.mode == 'test':
            solver.test()
    elif config.type == "multiclass":
        multisolver = MultiSolver(config, train_loader, valid_loader, test_loader, classes=classes)
        if config.mode == 'train':
            multisolver.train_model()
        elif config.mode == 'test':
            multisolver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--type', type=str, default="multiclass")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')  
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='femoral_d_U_Net-150-0.0004-11-0.4000.pkl')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\train')
    parser.add_argument('--valid_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\validation')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\ek779475\\Documents\\Koutoulakis\\automatic_segmentation\\Dataset\\multiclass\\test')
    parser.add_argument('--result_path', type=str, default='./result/')
    # To pass an list argument, you should type
    # i.e. python main.py --classes RECTUM VESSIE TETE_FEMORALE_D TETE_FEMORALE_G
    parser.add_argument('--classes', nargs="+", default=["BACKGROUND", "RECTUM","VESSIE","TETE_FEMORALE_D", "TETE_FEMORALE_G"], help="Be sure the you specified the classes to the exact order")

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    try:
        main(config)
    except KeyboardInterrupt:
        print("Keyboard Interruption")
        exit()