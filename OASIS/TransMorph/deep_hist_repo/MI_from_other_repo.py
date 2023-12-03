import os
import numpy as np

import torch
import torch.nn as nn

import skimage.io
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# from sklearn.metrics import normalized_mutual_info_score
"""Taken from https://github.com/connorlee77/pytorch-mutual-information"""

class MutualInformationFromOtherRepo(nn.Module):

	def __init__(self, sigma=0.1, num_bins=256, normalize=True, calc_mi_per_channel: bool = False,
				 subset_sample_for_histogram_build: bool = False, average_channels: bool = True):
		super(MutualInformationFromOtherRepo, self).__init__()

		self.sigma = sigma
		self.num_bins = num_bins
		self.normalize = normalize
		self.epsilon = 1e-10
		# my additions:
		self._calc_mi_per_channel = calc_mi_per_channel
		self._subset_sample_for_histogram_build = subset_sample_for_histogram_build  # this throw away 90% of the data to build the histogram
		self._average_channels = average_channels


	def forward(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''
		return self.getMutualInformation(input1, input2)

	def marginalPdf(self, values):
		if self._calc_mi_per_channel:
			residuals = (values - self.bins).transpose(0, 2)
		else:
			residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
		pdf = pdf / normalization
		
		return pdf, kernel_values


	def jointPdf(self, kernel_values1, kernel_values2):

		joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
		normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
		pdf = joint_kernel_values / normalization

		return pdf


	def getMutualInformation(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''

		#make sure input1 and input2 are in [0,1]
		# input1 = (input1 - input1.min()) / (input1.max() - input1.min())
		# input2 = (input2 - input2.min()) / (input2.max() - input2.min())
		# Torch tensors for images between (0, 1)

		input1 = input1*255
		input2 = input2*255
		# input1 = input1.cpu()
		# input2 = input2.cpu()

		# mutual_information_gt = self._get_real_mutual_info(input1, input2)

		if len(input1.shape) == 5:  # 3D scans
			# to save memory, we need to average channels. x is 160x192x224, every 5 slices, we average them. so x will be 32x192x224
			# x shape (1,1,160,192,224)
			# average every 5 slices
			if self._average_channels:
				n_channels_avg = 5
				output_channels = input1.shape[2] // n_channels_avg
				input1 = torch.mean(input1.view(input1.shape[0], input1.shape[1], output_channels, n_channels_avg, input1.shape[3], input1.shape[4]), dim=3)
				input2 = torch.mean(input2.view(input2.shape[0], input2.shape[1], output_channels, n_channels_avg, input2.shape[3], input2.shape[4]), dim=3)

			input1 = input1[:, 0]
			input2 = input2[:, 0]
			B, C, H, W = input1.shape
			x1 = input1.view(B, H * W * C, 1)  # 1 is the number of channels
			x2 = input2.view(B, H * W * C, 1)
			if self._subset_sample_for_histogram_build:
				# x1 x2 is (1,N,1). we want to take 10% of N (randomly) so that x1 x2 will be (1,0.1N,1)
				x1 = x1[:, torch.randperm(x1.shape[1])[:int(0.5*x1.shape[1])], :]
				x2 = x2[:, torch.randperm(x2.shape[1])[:int(0.5*x2.shape[1])], :]


		else:
			B, C, H, W = input1.shape
			x1 = input1.view(B, H * W, C)
			x2 = input2.view(B, H * W, C)
		assert((input1.shape == input2.shape))

		if self._calc_mi_per_channel:
			self.bins = torch.linspace(0, 255, self.num_bins, dtype=input1.dtype).to(input1.device)
			self.bins = self.bins.view(self.num_bins, 1, 1).repeat(1, H * W, C)  # Reshape bins to (256, N, C)
		else:
			self.bins = nn.Parameter(torch.linspace(0, 255, self.num_bins).float(), requires_grad=False).to(input1.device)
		# and this is marginalPdf:

		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)
		pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

		H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
		H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
		H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

		mutual_information = H_x1 + H_x2 - H_x1x2
		
		if self.normalize:
			# GithubCopilot says: what it does is to normalize the mutual information by the average entropy of the two images
			mutual_information = 2*mutual_information/(H_x1+H_x2)

		return mutual_information.mean()

	def _get_real_mutual_info(self, input1, input2):
		input1_np = input1[0, 0].cpu().numpy().flatten()
		input2_np = input2[0, 0].cpu().numpy().flatten()
		from sklearn.metrics import normalized_mutual_info_score
		return normalized_mutual_info_score(input1_np, input2_np)


if __name__ == '__main__':

	device = 'cuda:0'

	### Create test cases ###
	img1 = Image.open('grad.jpg').convert('L')
	img2 = img1.rotate(10)

	arr1 = np.array(img1)
	arr2 = np.array(img2)

	mi_true_1 = normalized_mutual_info_score(arr1.ravel(), arr2.ravel())
	mi_true_2 = normalized_mutual_info_score(arr2.ravel(), arr2.ravel())

	img1 = transforms.ToTensor() (img1).unsqueeze(dim=0).to(device)
	img2 = transforms.ToTensor() (img2).unsqueeze(dim=0).to(device)

	# Pair of different images, pair of same images
	input1 = torch.cat([img1, img2])
	input2 = torch.cat([img2, img2])

	MI = MutualInformationFromOtherRepo(num_bins=256, sigma=0.1, normalize=False).to(device)
	mi_test = MI(input1, input2)

	mi_test_1 = mi_test[0].cpu().numpy()
	mi_test_2 = mi_test[1].cpu().numpy()

	print('Image Pair 1 | sklearn MI: {}, this MI: {}'.format(mi_true_1, mi_test_1))
	print('Image Pair 2 | sklearn MI: {}, this MI: {}'.format(mi_true_2, mi_test_2))

	# assert(np.abs(mi_test_1 - mi_true_1) < 0.05)
	# assert(np.abs(mi_test_2 - mi_true_2) < 0.05)

	## my:
	from mor_mimic.hist_layers import JointHistLayer
	from mor_mimic.metrics import MutualInformationLoss

	channel_to_take = 0
	xy_joint_histogram, x_histogram, y_histogram = JointHistLayer()(img1[:, channel_to_take], img2[:, channel_to_take])
	mor_mi = MutualInformationLoss()(x_histogram, y_histogram, xy_joint_histogram)
	print(f"Mor Mutual Information Loss: {mor_mi}")
