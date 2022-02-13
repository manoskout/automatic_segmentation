import os
from cv2 import drawContours
from matplotlib.pyplot import draw
import numpy as np
import time
import datetime
from sklearn.metrics import f1_score, jaccard_score
import torch
# Fix memory problem
torch.cuda.empty_cache()
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from tqdm import tqdm
import cv2

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
		self.augmentation_prob = config.augmentation_prob
		self.min_valid_loss = np.inf					

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.amp = False # Pass it as config value


		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

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

		self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()


	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img

	def getContours(im):
		imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray,127,255,0)
		contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		return contours
	def draw_contours(image,predicted, true_mask):
		"""
		This function draws both the predicted mask and ground truth mask
		The green contours are the predicted
		The blue contours are the ground truth
		"""
		pred_contours = getContours(predicted)
		true_contours = getContours(true_mask)

		with_pred = cv2.drawContours(image, pred_contours,-1,(0,255,0),1)
		combined_img = cv2.drawContours(with_pred, true_contours,-1,(255,0,0),1)
		
		return combined_img



	def save_validation_results(self, image, pred_mask, true_mask,epoch):
		image = image.data.cpu().numpy()
		pred_mask = pred_mask.data.cpu().numpy()
		true_mask = true_mask.data.cpu().numpy()

		final_img = drawContours(image,pred_mask,true_mask)
		torchvision.utils.save_image(
			final_img,
			os.path.join(
				self.result_path,
				'%s_valid_%d_result.png'%(self.model_type,epoch+1
				)
			)
		)


	

	def evaluate(self,epoch):
		"""
		"""
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		valid_loss =0
		self.unet.eval()
		dice_c= f1_s = jaccard_s = 0.
		length = 0 
		num_val_batches = len(self.valid_loader)		
		for (image, true_mask) in tqdm(self.valid_loader, total = num_val_batches, desc="Validation Round", unit="batch", leave=False):
			image = image.to(self.device,dtype=torch.float32)
			true_mask = true_mask.to(self.device, dtype=torch.float32)

			with torch.no_grad():		
				pred_mask = self.unet(image)
				loss = self.criterion(pred_mask,true_mask)
				valid_loss = loss.item() * (image.size(0)/self.batch_size)
			length += image.size(0)/self.batch_size
		self.unet.train()
		dice_c += dice_coeff(pred_mask,true_mask)
		f1_s =0
		jaccard_s += jaccard_score(pred_mask, true_mask)

		# Save the prediction
		self.save_validation_results(image, pred_mask, true_mask,epoch)
		if self.min_valid_loss > valid_loss:
			print(f'Validation Loss Decreased({self.min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
			self.min_valid_loss = valid_loss
			# Saving State Dict
			torch.save(self.unet.state_dict(), unet_path)
		if num_val_batches == 0:
			return dice_c, f1_s, jaccard_s,valid_loss
		return dice_c, f1_s, jaccard_s,valid_loss
	def train_model(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		training_log = open(os.path.join(self.result_path,'result_train.csv'), 'a', encoding='utf-8', newline='')
		validation_log = open(os.path.join(self.result_path,'result_validation.csv'), 'a', encoding='utf-8', newline='')
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
			# print("n_train: ",n_train)
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				jaccard_s = dice_c = f1_s = 0.	
				length = 0
				with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='img') as pbar:
					for i, (images, true_masks) in enumerate(self.train_loader):

						import matplotlib.pyplot as plt
						images = images.to(self.device,dtype=torch.float32)
						true_masks = true_masks.to(self.device, dtype=torch.float32)
						# fig, (ax1,ax2) = plt.subplots(1,2)
						# ax1.imshow(torch.squeeze(images))
						# ax2.imshow(torch.squeeze(true_masks))
						# plt.show()
						assert images.shape[1] == self.img_ch, \
                    f'Network has been defined with {self.img_ch} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
						
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

						jaccard_s += jaccard_score(pred_mask,true_masks)
						dice_c += dice_coeff(pred_mask,true_masks)
						length += images.size(0)/self.batch_size
						pbar.set_postfix(**{'loss (batch)': loss.item()})


				jaccard_s = jaccard_s/length
				dice_c = dice_c/length
				f1_s = f1_s/length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] JS: %.4f, DC: %.4f, F1: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss/n_train,\
					  jaccard_s,dice_c,f1_s))
				# Update csv
				wr_train.writerow([jaccard_s,dice_c,f1_s,self.lr, epoch])

				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
				#===================================== Validation ====================================#
				dice_c = f1_c = jaccard_s = 0
				division_step = (n_train // (10 * self.batch_size))
				if division_step > 0:
					if global_step % division_step == 0:	
						self.unet.train(False)
						dice_c, f1_s, jaccard_s = self.evaluate(epoch)
				# print('[Validation] SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(SE,SP,PC,F1,JS,DC))
				print('[Validation] --> JS: %.4f, DC: %.4f, F1: %.4f'%(jaccard_s, dice_c, f1_s))
				wr_valid.writerow([jaccard_s,dice_c,f1_s,self.lr, epoch])

					
			# #===================================== Test ====================================#
			# del self.unet
			# self.build_model()
			# self.unet.load_state_dict(torch.load(unet_path))
			
			# self.unet.train(False)
			# self.unet.eval()
			# test_len = len(self.test_loader)
			# JS = 0.		# Jaccard Similarity
			# DC = 0.		# Dice Coefficient
			# for i, (images, true_masks) in enumerate(self.test_loader):

			# 	images = images.to(self.device)
			# 	true_masks = true_masks.to(self.device)
			# 	SR = torch.sigmoid(self.unet(images))
			# 	JS += get_JS(SR,true_masks)
			# 	DC += dice_coeff(SR,true_masks)
						
					
			# JS = JS/test_len
			# DC = DC/test_len
			# unet_score = JS + DC


			training_log.close()
			validation_log.close()
			

			
