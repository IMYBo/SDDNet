#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT
from dnnet import DN_Net
from srnet import SR_Net
from loss import si_snr, sd_snr, GD, kl, mix

class SDDNet(nn.Module):
	"""docstring for SDDNet"""
	def __init__(self, win_len=320, win_inc=160, fft_len=320, win_type='hanning', stage='DRNet'):
		"""
		stage: DNNet, DRNet, SRNet
		"""
		super(SDDNet, self).__init__()
		self.stage = stage
		self.dnnet = DN_Net(win_len=win_len, win_inc=win_inc, fft_len=fft_len, win_type=win_type, I=5, dereverb=False)
		if stage == 'DRNet':
			for param in self.parameters():
				param.requires_grad = False
		self.drnet = DN_Net(win_len=win_len, win_inc=win_inc, fft_len=fft_len, win_type=win_type, I=5, dereverb=True)
		if stage == 'SRNet':
			for param in self.parameters():
				param.requires_grad = False
		self.srnet = SR_Net(win_len=win_len, win_inc=win_inc, fft_len=fft_len, win_type=win_type)

		self.stft = ConvSTFT(win_len=win_len, win_inc=win_inc, fft_len=fft_len, win_type=win_type, feature_type='complex')
		self.istft = ConviSTFT(win_len=win_len, win_inc=win_inc, fft_len=fft_len, win_type=win_type, feature_type='complex')
		show_model(self)
		show_params(self)

	def forward(self, inputs, labels=None):
		"""
		inputs: B x T
		"""
		inputs_cspecs = self.stft(inputs)#B x F x T
		B, D, T = inputs_cspecs.size()
		inputs_phase = torch.atan2(inputs_cspecs[:, D//2:], inputs_cspecs[:, :D//2] + 1e-8)
		inputs_specs = torch.sqrt(inputs_cspecs[:, D//2:]**2 + inputs_cspecs[:, :D//2]**2)**0.5
		outs_specs = self.dnnet(inputs_specs)
		outs_phase = inputs_phase
		if self.stage == 'DRNet' or self.stage == 'SRNet':
			denoise_specs = outs_specs
			outs_specs = self.drnet(inputs_specs, denoise_specs)###去混模块线性谱要压缩

		if self.stage == 'SRNet':
			denoise_cspecs = torch.cat([denoise_specs*torch.cos(inputs_phase), denoise_specs*torch.sin(inputs_phase)], 1)
			dereverb_cspecs = torch.cat([outs_specs*torch.cos(inputs_phase), outs_specs*torch.sin(inputs_phase)], 1)
			inputs_cspecs = torch.cat([inputs_specs*torch.cos(inputs_phase), inputs_specs*torch.sin(inputs_phase)], 1)
			outs = self.srnet(inputs_cspecs, dereverb_cspecs, denoise_cspecs)
			real, imag = torch.chunk(outs, 2, 1)
			outs_specs = torch.sqrt(real**2 + imag**2 + 1e-8)
			outs_phase = torch.atan2(imag, real + 1e-8)

		
		raw_wav = self.istft(torch.cat([(outs_specs**2)*torch.cos(outs_phase), (outs_specs**2)*torch.sin(outs_phase)], 1))##wav

		outs_cspecs = torch.cat([(outs_specs)*torch.cos(outs_phase), (outs_specs)*torch.sin(outs_phase)], 1)
		raw_wav = torch.squeeze(raw_wav, 1)


		if labels is not None:
			gth_sepc = self.stft(labels)
			B, D, T = gth_sepc.size()
			gth_sepc[:, 0] = 0
			gth_sepc[:, D//2] = 0
			if self.stage == 'DNNet':
				mag_labels = torch.sqrt(gth_sepc[:, :D//2]**2 + gth_sepc[:, D//2:]**2)
			else:
				mag_labels = torch.sqrt(gth_sepc[:, :D//2]**2 + gth_sepc[:, D//2:]**2)**0.5

			phase_labels = torch.atan2(gth_sepc[:, D//2:], gth_sepc[:, :D//2] + 1e-8)
			gth_csepc = torch.cat([mag_labels*torch.cos(phase_labels), mag_labels*torch.sin(phase_labels)], 1)
			return [outs_cspecs,gth_csepc], raw_wav
		else:
			return raw_wav

	def norm(self, inputs):
		inputs = inputs / torch.clamp(inputs**2, 1e-30) #* inputs.shape[1]
		return inputs

	def wav2f(self, inputs):
		cspecs = self.stft(inputs)
		B, D, T = cspecs.size()
		#cspecs[:, 0] = 0
		#cspecs[:, D//2] = 0
		mags = torch.sqrt(cspecs[:, D//2:]**2 + cspecs[:, :D//2]**2 + 1e-8)
		return mags, cspecs

	def sisnr(self, inputs, labels):
		return -(si_snr(inputs, labels))
	
	def loss(self, inputs, labels, complex=True, loss_mode='MSE'):

		if loss_mode == 'MSE':
			B,D,T = inputs.size()
			inputs_mags = torch.sqrt(inputs[:, :D//2]**2 + inputs[:, D//2:]**2 + 1e-8)
			labels_mags = torch.sqrt(labels[:, :D//2]**2 + labels[:, D//2:]**2 + 1e-8)
			inputs_cspecs = inputs
			labels_cspecs = labels
			loss_mag = F.mse_loss((inputs_mags), (labels_mags))
			loss_RI = F.mse_loss(labels_cspecs[:, :D//2], inputs_cspecs[:, :D//2]) + F.mse_loss(labels_cspecs[:, D//2:], inputs_cspecs[:, D//2:])
			if complex:
				loss = loss_RI + loss_mag
			else:
				loss = loss_mag
			#print(loss)
			return loss
		elif loss_mode == 'SI-SNR':
			#return -torch.mean(si_snr(inputs, labels))
			length = min(inputs.shape[1], labels.shape[1])
			return -(si_snr(inputs[:, :length], labels[:, :length]))
		elif loss_mode == 'rGKL':
			return kl(inputs, labels)
	
	def get_params(self, weight_decay=0.0):
			# add L2 penalty
		weights, biases = [], []
		for name, param in self.named_parameters():
			if 'bias' in name:
				biases += [param]
			else:
				weights += [param]
		params = [{
					 'params': weights,
					 'weight_decay': weight_decay,
				 }, {
					 'params': biases,
					 'weight_decay': 0.0,
				 }]
		return params

def l2_norm(s1, s2):
	#norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
	#norm = torch.norm(s1*s2, 1, keepdim=True)
	
	norm = torch.sum(s1*s2, -1, keepdim=True)
	return norm 
	
def si_snr(s1, s2, eps=1e-8):
	#s1 = remove_dc(s1)
	#s2 = remove_dc(s2)
	s1_s2_norm = l2_norm(s1, s2)
	s2_s2_norm = l2_norm(s2, s2)
	s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
	e_nosie = s1 - s_target
	target_norm = l2_norm(s_target, s_target)
	noise_norm = l2_norm(e_nosie, e_nosie)
	snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
	return torch.mean(snr)


if __name__ == '__main__':
	Net = SDDNet(stage='SRNet')
	inputs = torch.randn([1, 16000])
	labels = inputs
	[outs_cspecs,gth_csepc], raw_wav = Net(inputs, labels)
	print(Net.loss(outs_cspecs, gth_csepc, complex=True))
	#print(len(outs), outs[0].shape, outs[1].shape, outs[-1].shape)

