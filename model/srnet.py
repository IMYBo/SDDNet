#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft_new import ConvSTFT, ConviSTFT
import math
class RealConv2d(nn.Module):
	"""docstring for Conv2dReal"""
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(RealConv2d, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		padding = (self.padding[0], 0)
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
								stride=stride, padding=padding)
	def forward(self, inputs):
		"""
		inputs: B x C x F x T
		"""
		if self.padding[1] != 0:
			inputs = F.pad(inputs, [self.padding[1], 0, 0, 0]) # padding T
		return self.conv(inputs)

class RealConvtranspose2d(nn.Module):
	"""docstring for Conv2dReal"""
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(RealConvtranspose2d, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		padding = (self.padding[0], 0)
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
								stride=stride, padding=padding)
	def forward(self, inputs):
		"""
		inputs: B x C x F x T
		"""
		if self.padding[1] != 0:
			inputs = F.pad(inputs, [self.padding[1], 0, 0, 0]) # padding T
		return self.conv(inputs)

class Gated_D_Conv(nn.Module):
	"""docstring for Gated_D_Conv"""
	def __init__(self, channels, kernel_size, dilation):
		super(Gated_D_Conv, self).__init__()
		self.main_conv = nn.Sequential(
							nn.PReLU(64),
							nn.InstanceNorm1d(channels, affine=True),
							nn.ConstantPad1d([(kernel_size-1)*dilation, 0, 0, 0], 0),
							nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, dilation=dilation, bias=False),
						)
		self.gate_conv = nn.Sequential(
							nn.PReLU(64),
							nn.InstanceNorm1d(channels, affine=True),
							nn.ConstantPad1d([(kernel_size-1)*dilation, 0, 0, 0], 0),
							nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, dilation=dilation, bias=False),
							nn.Sigmoid()
						)
	def forward(self, inputs):
		outputs = self.main_conv(inputs) * self.gate_conv(inputs)
		return outputs

		

class DMG_TCM(nn.Module):
	"""docstring for TCM"""
	def __init__(self, in_channels, out_channels, dilation, kernel_size=5):
		super(DMG_TCM, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conpress_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
		self.primal_domain = Gated_D_Conv(out_channels, kernel_size, dilation)
		self.dual_domain = Gated_D_Conv(out_channels, kernel_size, int(2**(5 - math.log2(dilation))))
		self.out_conv =	nn.Sequential(
							nn.PReLU(64*2),
							nn.InstanceNorm1d(out_channels*2, affine=True),
							nn.Conv1d(in_channels=out_channels*2, out_channels=in_channels, kernel_size=1, bias=False),
							)
		self.dilation = dilation
		self.kernel_size = kernel_size
	def forward(self, inputs):
		"""
		inputs: B x C x T
		outputs: B x C x T
		"""
		outputs = self.conpress_conv(inputs)
		outputs = torch.cat([self.primal_domain(outputs), self.dual_domain(outputs)], 1)
		#outputs = self.primal_domain(outputs) #torch.cat([self.primal_domain(outputs), self.dual_domain(outputs)], 1)
		outputs = self.out_conv(outputs)
		return outputs + inputs

class TcmBlocks(nn.Module):
	"""docstring for TcmBlocks"""
	def __init__(self, in_channels=256, hidden_channels=64, dilation=[1, 2, 4, 8, 16, 32]):
		super(TcmBlocks, self).__init__()
		self.TCMBlocks = nn.ModuleList()
		for i in range(len(dilation)):
			#if i == 0:
			self.TCMBlocks.append(DMG_TCM(in_channels=in_channels, out_channels=hidden_channels, dilation=dilation[i]))
			#elif i != len(dilation)-1:
			#	self.TCMBlocks.append(TCM(in_channels=hidden_channels, out_channels=hidden_channels, dilation=dilation[i]))
			#else:
			#	self.TCMBlocks.append(TCM(in_channels=hidden_channels, out_channels=in_channels, dilation=dilation[i]))

	def forward(self, inputs):
		"""
		inputs: B x C  x T
		outputs: B x C x T
		"""
		for i in range(len(self.TCMBlocks)):
			inputs = self.TCMBlocks[i](inputs)
		return inputs
		
class SR_Net(nn.Module):
	"""
	复数谱谱降噪网络
	"""
	def __init__(self, win_len=320, win_inc=160, fft_len=320, win_type='hanning'):
		super(SR_Net, self).__init__()
		self.encoder = nn.ModuleList()
		self.real_decoder = nn.ModuleList()
		self.imag_decoder = nn.ModuleList()
		self.tcmlayer_num = 3
		self.tcmlayer = nn.ModuleList()
		dilation = [1, 2, 4, 8, 16, 32]
		kernel_num = [6, 64, 64, 64, 64, 64]
		#self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, 'complex', fix=fix)
		#self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type, 'complex', fix=fix)
		for i in range(len(kernel_num) - 1):
			self.encoder.append(
					nn.Sequential(
						RealConv2d(
							in_channels=kernel_num[i],
							out_channels=kernel_num[i+1],
							kernel_size=(5, 2) if i == 0 else (3, 2),#F, T
							stride=(2, 1),
							padding=(0, 1)),
						nn.InstanceNorm2d(kernel_num[i+1]),
						nn.PReLU(),
					)
				)
		for i in range(self.tcmlayer_num):
				self.tcmlayer.append(TcmBlocks(dilation=dilation))
		for i in range(len(kernel_num)-1, 0, -1):
			self.real_decoder.append(
					nn.Sequential(
						nn.ConvTranspose2d(
							in_channels=kernel_num[i],
							out_channels=kernel_num[i-1] if i != 1 else 1,
							kernel_size=(3, 2) if i != 1 else (5, 2),
							stride=(2, 1),
							#padding=(0, 1),
						),
						nn.InstanceNorm2d(kernel_num[i-1]) if i != 1 else nn.InstanceNorm2d(1), 
						nn.PReLU()
					)
				)
			self.imag_decoder.append(
					nn.Sequential(
						nn.ConvTranspose2d(
							in_channels=kernel_num[i],
							out_channels=kernel_num[i-1] if i != 1 else 1,
							kernel_size=(3, 2) if i != 1 else (5, 2),
							stride=(2, 1),
							#padding=(0, 1),
						),
						nn.InstanceNorm2d(kernel_num[i-1]) if i != 1 else nn.InstanceNorm2d(1), 
						nn.PReLU()
					)
				)

	def multiframe_filtering(self, inputs, mask):
		"""
		inputs: B x I x F x T
		mask: B x I x F x T
		"""
		outputs = []
		for i in range(self.I):
			outputs.append(F.pad(inputs, [i, 0, 0, 0])[:, :, :-i] if i != 0 else inputs)
		#print(len(outputs), outputs[0].shape)
		outputs = torch.stack(outputs, 1)##B xI x F x T
		#print(outputs.shape)
		#print(mask.shape)
		outputs = outputs * mask
		return torch.sum(outputs, 1)
	def forward(self, inputs, dereverb_out, denoise_out):
		"""
		inputs: B x F x T
		"""
		#cspecs = self.stft(inputs)##B x (2F) x T
		#real, imag = torch.chunk(cspecs, 2, 1)
		outs = torch.chunk(inputs, 2, 1) + torch.chunk(dereverb_out, 2, 1) + torch.chunk(denoise_out, 2, 1)
		outs = torch.stack(outs, 1) # B x C x F x T
		encoder_out = []
		for i in range(len(self.encoder)):
			outs = self.encoder[i](outs)
			#print(f"encoder layer {i} : shape {outs.shape}")
			encoder_out.append(outs)

		B, C, D, T = outs.size()
		outs = outs.view(B, -1, T)
		for i in range(len(self.tcmlayer)):
			outs = self.tcmlayer[i](outs)

		outs = outs.view(B, C, D, T)
		#outs_real, outs_imag = torch.chunk(outs, 2, 1)
		#print(outs.shape)
		outs_real, outs_imag = outs, outs
		for i in range(len(self.real_decoder)):
			#real_encoder, imag_encoder = torch.chunk(encoder_out[-1-i], 2, 1)
			outs_real = self.real_decoder[i](outs_real + encoder_out[-1-i])
			outs_real = outs_real[..., :-1]
			outs_imag = self.imag_decoder[i](outs_imag + encoder_out[-1-i])
			outs_imag = outs_imag[..., :-1]
			#print(f"decoder layer {i} : shape {outs_imag.shape}")
		outs = dereverb_out + torch.cat([outs_real.squeeze(1), outs_imag.squeeze(1)], 1)
		return outs

if __name__ == "__main__":
	Net = SR_Net()
	inputs = torch.randn([10, 161*2, 1000])
	outs = Net(inputs, inputs, inputs)
	print(outs.shape)



