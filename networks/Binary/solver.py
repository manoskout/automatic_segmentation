import os
import numpy as np
import torch
# Fix memory problem
torch.cuda.empty_cache()
import torchvision
from torch import optim
from utils_metrics import DiceBCELoss,DiceLoss,FocalLoss,AverageMeter
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = DiceLoss()
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
		# self.optimizer = optim.RMSprop(self.unet.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
		self.unet.to(self.device)

	def save_validation_results(self, image, pred_mask, true_mask,epoch):
		image = image.data.cpu()
		pred_mask = pred_mask.data.cpu()
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

	def _update_metricRecords(self,mode,metric):	

		self.writer.add_scalars("loss", {mode:metric.avg_loss}, self.epoch)
		self.writer.add_scalars("recall", {mode:metric.avg_recall}, self.epoch)
		self.writer.add_scalars("sensitivity", {mode:metric.avg_sensitivity}, self.epoch)
		self.writer.add_scalars("specificity", {mode:metric.avg_specificity}, self.epoch)
		self.writer.add_scalars("dice", {mode:metric.avg_dice}, self.epoch)
		self.writer.add_scalars("jaccard", {mode:metric.avg_iou}, self.epoch)
		self.writer.add_scalars("hausdorff", {mode:metric.avg_hausdorff}, self.epoch)
		self.writer.add_scalars("hausforff_95", {mode:metric.avg_hd95}, self.epoch)
		self.wr_valid.writerow(
			[
				self.epoch+1,self.lr, metric.avg_loss, 
				metric.avg_precision, metric.avg_recall, metric.avg_sensitivity, 
				metric.avg_specificity, metric.avg_dice, metric.avg_iou,
				metric.avg_hausdorff,metric.avg_hd95
			])

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
			true_mask = true_mask.to(self.device, dtype=torch.float32)
			with torch.no_grad():		
				pred_mask = self.unet(image)
				loss = self.criterion(pred_mask,true_mask)
			metrics.update(loss.item(), true_mask, pred_mask, image.size(0)/self.batch_size)
			
		self.unet.train()
		# Save the prediction
		self.save_validation_results(image, pred_mask, true_mask,self.epoch)
		if self.min_valid_loss > metrics.avg_loss:
			print(f'[Validation] Loss Decreased({self.min_valid_loss:.6f}--->{metrics.avg_loss:.6f}) \t Saving The Model')
			self.min_valid_loss = metrics.avg_loss
			# Saving State Dict
			torch.save(self.unet.state_dict(), unet_path)
		print(f'[Validation] --> Epoch [{self.epoch+1}/{self.num_epochs}], Loss: {metrics.avg_loss}, DC: {metrics.avg_dice}, \
			Recall: {metrics.avg_recall}, Precision: {metrics.avg_precision}, Specificity: {metrics.avg_specificity}, \
			Sensitivity: {metrics.avg_sensitivity}, IoU: {metrics.avg_iou} , HD: {metrics.avg_hausdorff}, HD95: {metrics.avg_hd95}')
		self._update_metricRecords("Validation",metrics)
	
	def train_epoch(self):
		self.unet.train(True)
		metrics = AverageMeter()
		epoch_loss_values = list()
		with tqdm(total=self.n_train, desc=f'Epoch {self.epoch + 1}/{self.num_epochs}', unit='img') as pbar:
			for i, (image, true_mask) in enumerate(self.train_loader):
				image = image.to(self.device,dtype=torch.float32)
				true_mask = true_mask.to(self.device, dtype=torch.float32)

				assert image.shape[1] == self.img_ch, \
			f'Network has been defined with {self.img_ch} input channels'
				self.optimizer.zero_grad(set_to_none=True)
				# with torch.cuda.amp.autocast(enabled=self.amp):
				pred_mask = self.unet(image)
				loss = self.criterion(pred_mask,true_mask)

				# Backprop + optimize
				loss.backward()
				self.optimizer.step()

				pbar.update(int(image.shape[0]/self.batch_size))
				self.global_step +=1
				metrics.update(loss.item(), true_mask, pred_mask, image.size(0)/self.batch_size)

				pbar.set_postfix(**{'loss (batch)': loss.item()})
			epoch_loss_values.append(metrics.avg_loss)

	# Print the log info
		print(f'[Training] [{self.epoch+1}/{self.num_epochs}], Loss: {metrics.avg_loss}, DC: {metrics.avg_dice}, \
			Recall: {metrics.avg_recall}, Precision: {metrics.avg_precision}, Specificity: {metrics.avg_specificity}, \
			Sensitivity: {metrics.avg_sensitivity}, IoU: {metrics.avg_iou} , \
			HD: {metrics.avg_hausdorff}, HD95: {metrics.avg_hd95}')
		self._update_metricRecords("Training",metrics)

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
		
		self.wr_valid.writerow(["epoch","lr", "loss", "precision", "recall", "sensitivity", "specificity", "dice", "iou","hausdorff_distance","hausdorff_distance_95"])
		self.wr_train.writerow(["epoch","lr", "loss", "precision", "recall", "sensitivity", "specificity", "dice", "iou","hausdorff_distance","hausdorff_distance_95"])
		

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			print("The training process starts...")
			lr = self.lr
			self.n_train = len(self.train_loader)
			self.global_step = 0
			self.writer = SummaryWriter()
			# print("n_train: ",n_train)
			for epoch in range(self.num_epochs):
				self.epoch = epoch
				self.train_epoch()
				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					self.lr = lr # I did not update the rearning rate decay in other experiments
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				division_step = (self.n_train // (10 * self.batch_size))
				if division_step > 0:
					# print(f"Global Step : {self.global_step}, division_step: {division_step} Modulo: {self.global_step % division_step}")
					# if self.global_step % division_step == 0:	
					self.evaluation()
			training_log.close()
			validation_log.close()
