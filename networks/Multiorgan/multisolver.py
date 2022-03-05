import os
import numpy as np
import torch
# Fix memory problem
torch.cuda.empty_cache()
import torchvision
from torch import optim
from utils_metrics import AverageMeter
from losses import DiceLoss
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class MultiSolver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader, classes):

		# Data loader

		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		self.classes = classes
		del self.classes[0] # Delete the background label
		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		# Using this loss we dont have to perform one_hot is already implemented inside the function
		# self.criterion = torch.nn.CrossEntropyLoss()
		self.criterion = DiceLoss(mode=config.type)  
		  
		self.min_valid_loss = np.inf	
		self.model_name = config.model_name				

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.amp = False # Pass it as config value

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size
		
		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=self.img_ch,output_ch=self.output_ch)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=self.img_ch,output_ch=self.output_ch)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
			
		# self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									 self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

	def classes_to_mask(self,mask):
		"""Converts the labeled pixels to range 0-255
		"""
		for index, k in enumerate(self.classes):
			mask[mask==index] = int(255/index) if index != 0 else 0


		return mask.type(torch.float)

	def give_colour(out):  # --> Not used
		class_to_color = [torch.tensor([1.0, 0.0, 0.0]), ...]
		output = torch.zeros(1, 3, out.size(-2), out.size(-1), dtype=torch.float)
		for class_idx, color in enumerate(class_to_color):
			mask = out[:,class_idx,:,:] == torch.max(out, dim=1)[0]
			mask = mask.unsqueeze(1) # should have shape 1, 1, 100, 100
			curr_color = color.reshape(1, 3, 1, 1)
			segment = mask*color # should have shape 1, 3, 100, 100
			output += segment
		return output

	def save_validation_results(self, image, pred_mask, true_mask,epoch):
		# print(f"[BEFORE]   unique pred: {torch.unique(pred_mask)}, unique true: {torch.unique(true_mask)}")
		# print(f"image : {image.shape} , pred_masks : {pred_mask.shape}, true_mask: {true_mask.shape}")

		if len(self.classes)>1:
			pred_mask = torch.argmax(pred_mask,dim=1)
			pred_mask = self.classes_to_mask(pred_mask)
			# true_mask =torch.argmax(true_mask,dim=1)
			true_mask[0] = self.classes_to_mask(true_mask[0])
			true_mask = true_mask.to(torch.float32)
		# print(f"unique pred: {torch.unique(pred_mask)}, unique true: {torch.unique(true_mask)}")
		image = image.data.cpu()
		pred_mask = pred_mask.data.cpu()
		true_mask = true_mask.data.cpu()
		pred_mask.unsqueeze(1)
		# torch.set_printoptions(profile="full")
		# print(torch.unique(pred_mask))
		# print(f"image : {image.shape} , pred_masks : {pred_mask.shape}, true_mask: {true_mask.shape}")
		torchvision.utils.save_image(
			image,
			os.path.join(
				self.result_path,
				'%s_valid_%d_result_INPUT.jpg'%(self.model_type,epoch+1
				)
			)
		)
		torchvision.utils.save_image(
			pred_mask,
			os.path.join(
				self.result_path,
				'%s_valid_%d_result_PRED.jpg'%(self.model_type,epoch+1
				)
			)
		)
		torchvision.utils.save_image(
			true_mask,
			os.path.join(
				self.result_path,
				'%s_valid_%d_result_GT.jpg'%(self.model_type,epoch+1
				)
			)
		)

	def _update_metricRecords(self,csv_writer,mode,metric, classes=None):	
		avg_metrics = [
					self.epoch+1,self.lr, metric.avg_loss, 
					metric.all_precision, metric.all_recall, metric.all_sensitivity, 
					metric.all_specificity, metric.all_dice, metric.all_iou,
					metric.all_hd,metric.all_hd95
				]
		if classes:
			print(classes)
			for index in range(len(classes.items())):
				avg_metrics.append(metric.avg_iou[index])
				avg_metrics.append(metric.avg_dice[index])
				avg_metrics.append(metric.avg_hd[index])
			self.writer.add_scalars("loss", {mode:metric.avg_loss}, self.epoch)
			self.writer.add_scalars("recall", {mode:metric.all_recall}, self.epoch)
			self.writer.add_scalars("sensitivity", {mode:metric.all_sensitivity}, self.epoch)
			self.writer.add_scalars("specificity", {mode:metric.all_specificity}, self.epoch)
			self.writer.add_scalars("dice", {mode:metric.all_dice}, self.epoch)
			self.writer.add_scalars("jaccard", {mode:metric.all_iou}, self.epoch)
			self.writer.add_scalars("hausdorff", {mode:metric.all_hd}, self.epoch)
			self.writer.add_scalars("hausforff_95", {mode:metric.all_hd95}, self.epoch)
			csv_writer.writerow( 
				avg_metrics
				)
	


	def evaluation(self):
		"""
		"""
		self.unet.train(False)
		unet_path = os.path.join(
			self.model_path,
			'%s-%d-%.4f-%d.pkl'%(
				self.model_type,
				self.num_epochs,
				self.lr,
				self.num_epochs_decay,
				)
		)
		self.unet.eval()
		num_val_batches = len(self.valid_loader)
		metrics = AverageMeter()	
		for (image, true_mask) in tqdm(
			self.valid_loader, 
			total = num_val_batches, 
			desc="Validation Round", 
			unit="batch", 
			leave=False):

			image = image.to(self.device,dtype=torch.float32)
			true_mask = true_mask.to(self.device,dtype=torch.long)
			with torch.no_grad():		
				pred_mask = self.unet(image)
				if self.output_ch > 1:
					loss = self.criterion(pred_mask,true_mask[:,0,:,:])

				else:
					loss = self.criterion(pred_mask,true_mask)

			metrics.update(loss.item(), true_mask, pred_mask, image.size(0)/self.batch_size, classes=self.classes)
		self.unet.train()
		self.save_validation_results(image, pred_mask, true_mask,self.epoch)
		if self.min_valid_loss > metrics.avg_loss:
			print(f'[Validation] Loss Decreased({self.min_valid_loss:.6f}--->{metrics.avg_loss:.6f}) \t Saving The Model')
			self.min_valid_loss = metrics.avg_loss
			# Saving State Dict
			torch.save(self.unet.state_dict(), unet_path)
		print(f'[Validation] --> Epoch [{self.epoch+1}/{self.num_epochs}], Loss: {metrics.avg_loss}, DC: {metrics.all_dice}, \
			Recall: {metrics.all_recall}, Precision: {metrics.all_precision}, Specificity: {metrics.all_specificity}, \
			Sensitivity: {metrics.all_sensitivity}, IoU: {metrics.all_iou} , HD: {metrics.all_hd}, HD95: {metrics.all_hd95}')
		self._update_metricRecords(self.wr_valid,"Validation",metrics, self.classes)
		
	
	def train_epoch(self):
		self.unet.train(True)
		metrics = AverageMeter()

		epoch_loss_values = list()
		with tqdm(total=self.n_train, desc=f'Epoch {self.epoch + 1}/{self.num_epochs}', unit='img') as pbar:
			for i, (image, true_mask) in enumerate(self.train_loader):
				image = image.to(self.device,dtype=torch.float32)
				true_mask = true_mask.to(self.device, dtype=torch.long)

				assert image.shape[1] == self.img_ch, f'Network has been defined with {self.img_ch} input channels'
				self.optimizer.zero_grad(set_to_none=True)
				# with torch.cuda.amp.autocast(enabled=self.amp):
				pred_mask = self.unet(image)


				if self.output_ch > 1:
					loss = self.criterion(pred_mask,true_mask[:,0,:,:])
				else:
					loss = self.criterion(pred_mask,true_mask)
				# Backprop + optimize
				loss.backward()
				self.optimizer.step()
				pbar.update(int(image.shape[0]/self.batch_size))
				self.global_step +=1
				metrics.update(loss.item(), true_mask, pred_mask, image.size(0), classes=self.classes)

				pbar.set_postfix(**{'loss (batch)': loss.item()})
			epoch_loss_values.append(metrics.avg_loss)


		# Print the log info
		print(f'[Training] [{self.epoch+1}/{self.num_epochs}], Loss: {metrics.avg_loss}, DC: {metrics.all_dice}, \
			Recall: {metrics.all_recall}, Precision: {metrics.all_precision}, Specificity: {metrics.all_specificity}, \
			Sensitivity: {metrics.all_sensitivity}, IoU: {metrics.all_iou} , \
			HD: {metrics.all_hd}, HD95: {metrics.all_hd95}')

		self._update_metricRecords(self.wr_train,"Training",metrics, self.classes)

	def train_model(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		
		unet_path = os.path.join(
			self.model_path, 
			'%s-%d-%.4f-%d.pkl' %(
				self.model_type,
				self.num_epochs,
				self.lr,
				self.num_epochs_decay)
			)
		training_log = open(
			os.path.join(
				self.result_path,
				'result_train.csv'
			),
			'a', 
			encoding='utf-8', 
			newline=''
		)
		validation_log = open(
			os.path.join(
				self.result_path,
				'result_validation.csv'
			), 
			'a', 
			encoding='utf-8', 
			newline=''
		)

		self.wr_train = csv.writer(training_log)
		self.wr_valid = csv.writer(validation_log)
		
		metric_list = ["epoch","lr", "loss", "precision", "recall", "sensitivity", "specificity", "dice", "iou","hd","hd95"]
		for _, id in self.classes.items():
			for i in ["iou","dice","hd"]:
				metric_list.append(f"{id}_{i}")
		self.wr_valid.writerow(metric_list)
		self.wr_train.writerow(metric_list)
		

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			print("The training process starts...")
			self.n_train = len(self.train_loader)
			self.global_step = 0
			self.writer = SummaryWriter()
			for epoch in range(self.num_epochs):
				self.epoch = epoch
				self.train_epoch()
				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					self.lr -= (self.lr / float(self.num_epochs_decay))

					for param_group in self.optimizer.param_groups:
						param_group['lr'] = self.lr
					print ('Decay learning rate to lr: {}.'.format(self.lr))
				
				
				#===================================== Validation ====================================#
				division_step = (self.n_train // (10 * self.batch_size))
				if division_step > 0:
					# if self.global_step % division_step == 0:	
					self.evaluation()
			training_log.close()
			validation_log.close()
            
            
            
#------------------------------------------------------------------------------------------------------

import argparse
import os
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
