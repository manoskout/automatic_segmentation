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
from networks.losses import DiceLoss, FocalLoss # , DualFocalloss : lathos apotelesmata
# from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from networks.network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,ResAttU_Net
import csv
from tqdm import tqdm
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from Multiorgan.utils import build_model
from sklearn.model_selection import KFold


class MultiSolver(object):
	def __init__(
		self, config: argparse.Namespace, train_loader: data.DataLoader, valid_loader: data.DataLoader, 
		classes: dict, save_images: bool= False,) -> None:
		# K_fold Cross validation

		self.k_folds = config.k_folds
		# paths
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode
		self.log_dir = config.log_dir
		# Data loader

		self.train_loader = train_loader
		self.valid_loader = valid_loader
		if config.type == "multiclass":
			self.classes = classes
			try: 
				del self.classes[0] # Delete the background label
			except IndexError:
				print("Warning: there is no index 0 of the self.classes")
				print(self.classes)
				print("=====================PASS=====================")
				pass
		else:
			classes = []
		self.save_images = save_images
		# Model
		self.unet = None
		self.optimizer = None
		self.scheduler = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.dropout = config.dropout
		self.early_patience = config.early_stopping
		# Using this loss we dont have to perform one_hot is already implemented inside the function
		# self.criterion = torch.nn.CrossEntropyLoss()
		self.criterion = FocalLoss(mode=config.type, gamma=3) # TverskyLoss(mode=config.type, gamma= 3)  
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

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print("Device that is in use : {}".format(self.device))
		self.model_type = config.model_type
		self.t = config.t
		self.unet = build_model(config)
		self.optimizer = optim.Adam(
			list(self.unet.parameters()), self.lr,
			[self.beta1, self.beta2],weight_decay=1e-5)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,"min", patience=self.num_epochs_decay, 
			verbose=True,min_lr = 1e-6)

	
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
		pred_mask = pred_mask.data.cpu().squeeze()

		true_mask = true_mask.data.cpu()


		torchvision.utils.save_image(
			image,
			os.path.join(
				self.result_path,
				'%s_valid_%d_result_INPUT.png'%(self.model_type,epoch+1
				)
			)
		)
		torchvision.utils.save_image(
			pred_mask,
			os.path.join(
				self.result_path,
				'%s_valid_%d_result_PRED.png'%(self.model_type,epoch+1
				)
			)
		)
		torchvision.utils.save_image(
			true_mask,
			os.path.join(
				self.result_path,
				'%s_valid_%d_result_GT.png'%(self.model_type,epoch+1
				)
			)
		)

	def _update_metricRecords(self,csv_writer,mode,metric, classes=None, epoch=1, lr= None) -> None:	
		avg_metrics = [
					epoch+1,lr, metric.avg_loss, 
					metric.all_precision, metric.all_recall, metric.all_sensitivity, 
					metric.all_specificity, metric.all_dice, metric.all_iou,
					metric.all_hd,metric.all_hd95
				]
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
		if mode == "Training":
			self.writer.add_scalars("learning_rate", {mode:lr}, epoch)
		self.writer.add_scalars("loss", {mode:metric.avg_loss}, epoch)
		self.writer.add_scalars("recall", {mode:metric.all_recall}, epoch)
		self.writer.add_scalars("sensitivity", {mode:metric.all_sensitivity}, epoch)
		self.writer.add_scalars("specificity", {mode:metric.all_specificity}, epoch)
		self.writer.add_scalars("dice", {mode:metric.all_dice}, epoch)
		self.writer.add_scalars("jaccard", {mode:metric.all_iou}, epoch)
		self.writer.add_scalars("hausdorff", {mode:metric.all_hd}, epoch)
		self.writer.add_scalars("hausforff_95", {mode:metric.all_hd95}, epoch)
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
					loss = self.criterion(pred_mask,true_mask.squeeze(1))

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
		self._update_metricRecords(self.wr_valid,"Validation",metrics, self.classes, self.epoch, self.optimizer.param_groups[0]['lr'])
		avg_loss =metrics.avg_loss
		metrics.reset()
		return avg_loss
	
	def train_epoch(self) -> None:
		self.unet.train(True)
		metrics = AverageMeter()
		with tqdm(total=self.n_train, desc=f'Epoch {self.epoch + 1}/{self.num_epochs}', unit='img') as pbar:
			for i, (image, true_mask) in enumerate(self.train_loader):
				image = image.to(self.device,dtype=torch.float32)
				true_mask = true_mask.to(self.device, dtype=torch.long)
				self.optimizer.zero_grad()
				pred_mask = self.unet(image)

				if self.output_ch > 1:
					loss = self.criterion(pred_mask,true_mask.squeeze(1))
				else:
					loss = self.criterion(pred_mask,true_mask)
				# Backprop + optimize
				loss.backward()
				self.optimizer.step()
				pbar.update(int(image.shape[0]/self.batch_size))
				
				metrics.update(loss.item(), true_mask, pred_mask, image.size(0), classes=self.classes)

				pbar.set_postfix(**{'loss (batch)': loss.item()})
		# Print the log info
		print(f"[Training] [{self.epoch+1}/{self.num_epochs}, Lr: {self.optimizer.param_groups[0]['lr']}], Loss: {metrics.avg_loss}, DC: {metrics.all_dice}, \
			Recall: {metrics.all_recall}, Precision: {metrics.all_precision}, Specificity: {metrics.all_specificity}, \
			Sensitivity: {metrics.all_sensitivity}, IoU: {metrics.all_iou} , \
			HD: {metrics.all_hd}, HD95: {metrics.all_hd95}")

		self._update_metricRecords(self.wr_train,"Training",metrics, self.classes, self.epoch, self.optimizer.param_groups[0]['lr'])
		metrics.reset()
		#===================================== Validation ====================================#
		division_step = (self.n_train // (10 * self.batch_size))
		if division_step > 0:
			val_loss = self.evaluation()
			self.scheduler.step(val_loss)

		# TODO : Change this command below
		self.lr = self.optimizer.param_groups[0]['lr']
		self.early_stopping(val_loss, self.unet)
		
	def train_model(self,fold) -> None:
		"""Training"""
		# Set tensorboard writer
		self.writer = SummaryWriter(log_dir=self.log_dir+f"_fold_{fold}")		
		unet_path = os.path.join(
			self.result_path, 
			'%s-%d-%.4f-%d_%s.pkl' %(
				self.model_type,
				self.num_epochs,
				self.lr,
				self.num_epochs_decay,
				fold)
			)
		training_log = open(
			os.path.join(
				self.result_path,
				f'result_train_{fold}.csv'
			),
			'a', 
			encoding='utf-8', 
			newline=''
		)
		validation_log = open(
			os.path.join(
				self.result_path,
				f'result_validation_{fold}.csv'
			), 
			'a', 
			encoding='utf-8', 
			newline=''
		)
		self.wr_train = csv.writer(training_log)
		self.wr_valid = csv.writer(validation_log)

		self.early_stopping = EarlyStopping(patience=self.early_patience, verbose=True, path=unet_path)
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
            