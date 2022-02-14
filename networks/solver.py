import os
import numpy as np
import torch
# Fix memory problem
torch.cuda.empty_cache()
import torchvision
from torch import optim
from utils_metrics import *
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
		self.criterion = DiceBCELoss()
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
			
		self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
		self.optimizer = optim.Adam(list(self.unet.parameters()),
									 self.lr, [self.beta1, self.beta2])
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
		
	def evaluation(self,epoch, writer):
		"""
		"""
		unet_path = os.path.join(
			self.model_path, 
			'%s-%d-%.4f-%d.pkl'%(
				self.model_type,
				self.num_epochs,
				self.lr,
				self.num_epochs_decay,
				)
		)
		
		valid_loss =0
		self.unet.eval()
		length = 0 
		num_val_batches = len(self.valid_loader)		
		hd_distance = HausdorffDistance()
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
				valid_loss = loss.item() * (image.size(0)/self.batch_size)
			length += image.size(0)/self.batch_size
		self.unet.train()
		dice_c = dice_coeff(pred_mask,true_mask)
		hausdorff = hd_distance.compute(pred_mask,true_mask)
		# Save the prediction
		self.save_validation_results(image, pred_mask, true_mask,epoch)
		if self.min_valid_loss > valid_loss:
			print(f'[Validation] Loss Decreased({self.min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
			self.min_valid_loss = valid_loss
			# Saving State Dict
			torch.save(self.unet.state_dict(), unet_path)
		
		writer.add_scalar("val_mean_dice", dice_c, epoch + 1)
		return dice_c,valid_loss,hausdorff

	def train_model(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		
		unet_path = os.path.join(
			self.model_path, 
			'%s-%d-%.4f-%d.pkl' %(
				self.model_type,
				self.num_epochs,
				self.lr,
				self.num_epochs_decay				)
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

		wr_train = csv.writer(training_log)
		wr_valid = csv.writer(validation_log)
		

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			print("The training process starts...")
			# Train for Encoder
			lr = self.lr
			n_train = len(self.train_loader)
			global_step = 0
			writer = SummaryWriter()
			hd_distance = HausdorffDistance()
			# print("n_train: ",n_train)
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				dice_c = 0.	
				length = 0
				step = 0
				epoch_loss_values = list()
				with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img') as pbar:
					for i, (images, true_masks) in enumerate(self.train_loader):
						step+=1
						images = images.to(self.device,dtype=torch.float32)
						true_masks = true_masks.to(self.device, dtype=torch.float32)

						assert images.shape[1] == self.img_ch, \
                    f'Network has been defined with {self.img_ch} input channels'
						
						with torch.cuda.amp.autocast(enabled=self.amp):
							pred_mask = self.unet(images)
							loss = self.criterion(pred_mask,true_masks)
						

						# Backprop + optimize
						self.optimizer.zero_grad(set_to_none=True)
						self.grad_scaler.scale(loss).backward()
						self.grad_scaler.step(self.optimizer)
						self.grad_scaler.update()

						pbar.update(int(images.shape[0]/self.batch_size))
						global_step +=1
						epoch_loss += loss.item()
						hausdorff = hd_distance.compute(pred_mask,true_masks)
						dice_c += dice_coeff(pred_mask,true_masks)
						length += images.size(0)/self.batch_size
						writer.add_scalar("train_loss", loss.item(), self.num_epochs*epoch + step)
						pbar.set_postfix(**{'loss (batch)': loss.item()})
					epoch_loss /=step
					epoch_loss_values.append(epoch_loss)

				dice_c = dice_c/length

				# Print the log info
				print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss/n_train}, \n[Training] DC: {dice_c} Hausdorff : {hausdorff}')
				# Update csv
				wr_train.writerow([epoch+1,self.lr,epoch_loss/n_train, dice_c, hausdorff])

				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				dice_c = 0
				division_step = (n_train // (10 * self.batch_size))
				if division_step > 0:
					if global_step % division_step == 0:	
						self.unet.train(False)
						dice_c, valid_loss, hausdorff = self.evaluation(epoch, writer)
				print(f'[Validation] --> DC: {dice_c}, Loss: {valid_loss}, Hausdorff: {hausdorff}')
				wr_valid.writerow([epoch+1,self.lr,valid_loss, dice_c, hausdorff])


			training_log.close()
			validation_log.close()
			writer.close


	#===================================== Test ====================================#
	def test(self,):
		testing_log = open(
			os.path.join(
				self.result_path,
				'result_testing.csv'
			), 
			'a', 
			encoding='utf-8', 
			newline=''
		)
		wr_test = csv.writer(testing_log)

		unet_path = os.path.join(self.model_path, self.model_name)
		del self.unet
		self.build_model()
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		self.unet.eval()
		test_len = len(self.test_loader)
		hd_distance = HausdorffDistance()

		for i, (images, true_masks) in enumerate(self.test_loader):

			images = images.to(self.device)
			true_masks = true_masks.to(self.device)
			pred_masks = self.unet(images)
			dc = dice_coeff(pred_masks,true_masks)
			hausdorff = hd_distance(pred_masks,true_masks)
			wr_test.writerow([i, dc, hausdorff])

			self.save_validation_results(images, pred_masks, true_masks,test_len-i)
