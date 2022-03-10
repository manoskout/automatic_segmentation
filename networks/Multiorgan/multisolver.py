import argparse
import os
import numpy as np
import torch
# Fix memory problem
torch.cuda.empty_cache()
import torchvision
from torch import optim
from utils_metrics import AverageMeter, EarlyStopping
import segmentation_models_pytorch as smp 
from losses import DiceLoss
# from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net, SlaveAttU_Net
import csv
from tqdm import tqdm
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


class MultiSolver(object):
	def __init__(
		self, config: argparse.Namespace, train_loader: data.DataLoader, 
		valid_loader: data.DataLoader, 
		classes: list, save_images: bool= False) -> None:

		# Paths
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		# Data loader

		self.train_loader = train_loader
		self.valid_loader = valid_loader
		if config.type == "multiclass":
			self.classes = classes
			del self.classes[0] # Delete the background label
		else:
			classes = []
		self.save_images = save_images
		# Model
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.dropout = config.dropout
		# Using this loss we dont have to perform one_hot is already implemented inside the function
		# self.criterion = torch.nn.CrossEntropyLoss()
		self.criterion = DiceLoss(mode=config.type)  

		self.smp_enabled = config.smp
		self.encoder_name = config.encoder_name
		self.encoder_weights=config.encoder_weights		  
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
		
		
		# Set tensorboard writer
		self.writer = SummaryWriter(log_dir=config.log_dir)


		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print("Device that is in use : {}".format(self.device))
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		# TODO -> Dropout layers are not implemented into the R2U_net and R2Att_unet
		if self.smp_enabled:
			if self.model_type == "DeepLabV3+":
				self.unet = smp.DeepLabV3Plus(
					encoder_name=self.encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
					encoder_weights=self.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
					in_channels=self.img_ch,        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
					classes=self.output_ch,         # model output channels (number of classes in your dataset)
				)
			elif self.model_type == "U_Net":
				self.unet = smp.Unet(
					encoder_name=self.encoder_name,       
					encoder_weights=self.encoder_weights,    
					in_channels=self.img_ch,       
					classes=self.output_ch,         
				)
			elif self.model_type == "U_Net_plus":
				self.unet = smp.UnetPlusPlus(
					encoder_name=self.encoder_name,        
					encoder_weights=self.encoder_weights,    
					in_channels=self.img_ch,       
					classes=self.output_ch,         
				)
			# elif self.model_type =="":
			# 	self.unet = smp.
		elif self.model_type =='U_Net':
			self.unet = U_Net(img_ch=self.img_ch,output_ch=self.output_ch, dropout=self.dropout)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=self.img_ch,output_ch=self.output_ch, dropout=self.dropout)
		elif self.model_type =='SlaveAttU_Net':
			self.unet = SlaveAttU_Net(img_ch=self.img_ch,output_ch=self.output_ch, dropout=self.dropout)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=self.img_ch,output_ch=self.output_ch,t=self.t)
			
		# self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									 self.lr, [self.beta1, self.beta2],weight_decay=1e-5)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,"min", patience=self.num_epochs_decay, verbose=True,min_lr = 1e-6)
		self.unet.to(self.device)

	def classes_to_mask(self,mask : torch.Tensor) -> torch.Tensor:
		"""Converts the labeled pixels to range 0-255
		"""
		for index, k in enumerate(self.classes):
			mask[mask==index] = int(255/index) if index != 0 else 0


		return mask.type(torch.float)

	def save_validation_results(self, image, pred_mask, true_mask,epoch):

		if len(self.classes)>1:
			pred_mask = torch.argmax(pred_mask,dim=1)
			pred_mask = self.classes_to_mask(pred_mask)
			true_mask[0] = self.classes_to_mask(true_mask[0])
			true_mask = true_mask.to(torch.float32)
		image = image.data.cpu()
		pred_mask = pred_mask.data.cpu()
		true_mask = true_mask.data.cpu()
		pred_mask.unsqueeze(1)

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

	def _update_metricRecords(self,csv_writer,mode,metric, classes=None) -> None:	
		avg_metrics = [
					self.epoch+1,self.optimizer.param_groups[0]['lr'], metric.avg_loss, 
					metric.all_precision, metric.all_recall, metric.all_sensitivity, 
					metric.all_specificity, metric.all_dice, metric.all_iou,
					metric.all_hd,metric.all_hd95
				]
		if classes:
			for index in range(len(classes.items())):
				avg_metrics.append(metric.avg_iou[index])
				avg_metrics.append(metric.avg_dice[index])
				avg_metrics.append(metric.avg_hd[index])
		if mode == "Training":
			self.writer.add_scalars("learning_rate", {mode:self.optimizer.param_groups[0]['lr']}, self.epoch)
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
	
	def evaluation(self) -> float:
		"""
		"""
		self.unet.train(False)
		unet_path = os.path.join(
			self.result_path,
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
		if self.save_images:
			self.save_validation_results(image, pred_mask, true_mask,self.epoch)
		print('[Validation] --> Epoch [{epoch}/{epochs}], Loss: {avgLoss}, DC: {dc}, Recall: {recall}, Precision: {prec}, Specificity: {spec}, Sensitivity: {sens}, IoU: {iou} , HD: {hd}, HD95: {hd95}'
		.format(epoch=self.epoch+1, epochs=self.num_epochs,avgLoss = metrics.avg_loss, dc= metrics.all_dice,
		recall = metrics.all_recall, prec= metrics.all_precision, spec= metrics.all_specificity, sens = metrics.all_sensitivity,
		iou = metrics.all_iou, hd= metrics.all_hd, hd95=metrics.all_hd95))
		self._update_metricRecords(self.wr_valid,"Validation",metrics, self.classes)
		return metrics.avg_loss
	
	def train_epoch(self) -> None:
		self.unet.train(True)
		metrics = AverageMeter()
		epoch_loss_values = list()
		contain_slave = False # It should be changed!!!!!!!!! SOS
		with tqdm(total=self.n_train, desc=f'Epoch {self.epoch + 1}/{self.num_epochs}', unit='img') as pbar:
			for i, (image, true_mask) in enumerate(self.train_loader):
				if type(image) == list:
					slave = image[0].to(self.device,dtype=torch.float32)
					image = image[1].to(self.device,dtype=torch.float32)

				else:
					image = image.to(self.device,dtype=torch.float32)
				true_mask = true_mask.to(self.device, dtype=torch.long)
				assert image.shape[1] == self.img_ch, f'Network has been defined with {self.img_ch} input channels'
				self.optimizer.zero_grad(set_to_none=True)
				# with torch.cuda.amp.autocast(enabled=self.amp):
				if contain_slave:
					pred_mask = self.unet(image,slave)

				else:
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
		print(f"[Training] [{self.epoch+1}/{self.num_epochs}, Lr: {self.optimizer.param_groups[0]['lr']}], Loss: {metrics.avg_loss}, DC: {metrics.all_dice}, \
			Recall: {metrics.all_recall}, Precision: {metrics.all_precision}, Specificity: {metrics.all_specificity}, \
			Sensitivity: {metrics.all_sensitivity}, IoU: {metrics.all_iou} , \
			HD: {metrics.all_hd}, HD95: {metrics.all_hd95}")

		self._update_metricRecords(self.wr_train,"Training",metrics, self.classes)
		
		#===================================== Validation ====================================#
		division_step = (self.n_train // (10 * self.batch_size))
		if division_step > 0:
			# if self.global_step % division_step == 0:	
			val_loss = self.evaluation()
			self.scheduler.step(metrics.avg_loss)

		# TODO : Change this command below
		self.lr = self.optimizer.param_groups[0]['lr']
		self.early_stopping(val_loss, self.unet)
		
	def train_model(self) -> None:
		"""Training"""		
		unet_path = os.path.join(
			self.result_path, 
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

		self.early_stopping = EarlyStopping(patience=20, verbose=True, path=unet_path)
		val_loss = 0
		metric_list = ["epoch","lr", "loss", "precision", "recall", "sensitivity", "specificity", "dice", "iou","hd","hd95"]
		if self.classes:
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
			self.n_train = len(self.train_loader)
			self.global_step = 0
			for epoch in range(self.num_epochs):
				self.epoch = epoch
				self.train_epoch()				

				if self.early_stopping.early_stop:
					print("Early stopping")
					training_log.close()
					validation_log.close()
					break
			training_log.close()
			validation_log.close()
            