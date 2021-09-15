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


class smoothed_dilated_conv1d_GI(nn.Module):
	"""docstring for smoothed_dilated_conv1d_GI"""
	def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=True):
		super(smoothed_dilated_conv1d_GI, self).__init__()
		self.kernel_size = kernel_size
		self.dilation = dilation
		self.conv_block = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
		self.fix_w = nn.Parameter(torch.eye(dilation))
		self.bias = bias
		if bias:
			self.bias_block = nn.Parameter(torch.zeros(1, out_channels, 1), requires_grad=True)
	def forward(self, inputs):
		"""
		inputs: B x C x T
		"""
		#print('dilation', self.dilation)
		o_inputs = inputs
		#inputs = F.pad(inputs, [(self.dilation)*(self.kernel_size-1), 0, 0, 0])##pad
		B, C, T = inputs.size()
		#print('inputs1: ', inputs.shape)
		pad_t = (self.dilation - T % self.dilation) if T % self.dilation != 0 else 0
		inputs = F.pad(inputs, [pad_t, 0, 0, 0])
		B, C, T = inputs.size()
		#print('inputs2: ', inputs.shape)
		inputs = inputs.transpose(1, 2)##B x T x C
		inputs = inputs.contiguous().view(B * self.dilation, T//self.dilation, C).transpose(1, 2)##(B*d) x C x (T/d)
		inputs = F.pad(inputs, [(self.kernel_size-1), 0, 0, 0])##pad
		
		outs_conv = self.conv_block(inputs)
		#print('outs_conv: ', outs_conv.shape)
		outs_conv = torch.chunk(outs_conv, self.dilation, 0)
		out = []
		for i in range(self.dilation):
			out.append(self.fix_w[0, i] * outs_conv[i])
			for j in range(1, self.dilation):
				out[i] += self.fix_w[j, i] * outs_conv[i]

		out = torch.cat(out, 0)##(B*d) x C x (T/d)
		out = out.transpose(1, 2)
		out = out.contiguous().view(B, T, C)
		out = out[:, :o_inputs.shape[2], :].transpose(1, 2)
		if self.bias:
			out = out + self.bias_block

		return out

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
		#self.dual_domain = Gated_D_Conv(out_channels, kernel_size, int(2**(5 - math.log2(dilation))))
		self.out_conv = nn.Sequential(
							nn.PReLU(64),
							nn.InstanceNorm1d(out_channels, affine=True),
							nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias=False),
							)

		
		self.dilation = dilation
		self.kernel_size = kernel_size
	def forward(self, inputs):
		"""
		inputs: B x C x T
		outputs: B x C x T
		"""
		outputs = self.conpress_conv(inputs)
		outputs = self.primal_domain(outputs)#torch.cat([self.primal_domain(outputs), self.dual_domain(outputs)], 1)
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
		
class DN_Net(nn.Module):
	"""
	线性谱降噪去混网络
	"""
	def __init__(self, win_len=320, win_inc=160, fft_len=320, win_type='hanning', I=5, dereverb=False):
		super(DN_Net, self).__init__()
		#self.stft = ConvSTFT(win_len=win_len, win_inc=win_inc, fft_len=fft_len, win_type=win_type, feature_type='complex')
		#self.istft = ConviSTFT(win_len=win_len, win_inc=win_inc, fft_len=fft_len, win_type=win_type, feature_type='complex')
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()
		self.tcmlayer_num = 3
		self.dereverb = dereverb
		self.I = I
		self.tcmlayer = nn.ModuleList()
		dilation = [1, 2, 4, 8, 16, 32]
		kernel_num = [1, 64, 64, 64, 64, 64] if not dereverb else [2, 64, 64, 64, 64, 64]
		self.frq_dim = 161
		for i in range(len(kernel_num) - 1):
			self.encoder.append(
					nn.Sequential(
						nn.ConstantPad2d([1, 0, 0, 0], 0),
						nn.Conv2d(
							in_channels=kernel_num[i],
							out_channels=kernel_num[i+1],
							kernel_size=(5, 2) if i == 0 else (3, 2),#F, T
							stride=(2, 1),
							padding=(0, 0)),
						#print(self.frq_dim//(2**(i+1)) - 1)
						#nn.InstanceNorm2d([kernel_num[i+1], self.frq_dim//(2**(i+1)) - 1]),
						nn.InstanceNorm2d(kernel_num[i+1], affine=True),
						nn.PReLU(kernel_num[i+1]),
					)
				)
		for i in range(self.tcmlayer_num):
			self.tcmlayer.append(TcmBlocks(dilation=dilation))
		for i in range(len(kernel_num)-1, 0, -1):
			self.decoder.append(
					nn.Sequential(
						nn.ConvTranspose2d(
							in_channels=kernel_num[i],
							out_channels=kernel_num[i-1] if i != 1 else I,
							kernel_size=(3, 2) if i != 1 else (5, 2),
							stride=(2, 1),
							#padding=(0, 1),
						),
						nn.InstanceNorm2d(kernel_num[i-1], affine=True) if i != 1 else nn.InstanceNorm2d(I, affine=True), 
						nn.PReLU(kernel_num[i-1]) if i != 1 else nn.PReLU(I)
					)
				)
		show_model(self)
		show_params(self)
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
	def forward(self, inputs, denoise_out=None):
		"""
		inputs: B x F x T
		"""
		#cspecs = self.stft(inputs)##B x (2F) x T
		#real, imag = torch.chunk(cspecs, 2, 1)
		specs = inputs#torch.sqrt(real**2 + imag**2)
		#phase = torch.atan2(imag, real)
		if self.dereverb:
			outs = torch.stack([inputs, denoise_out], 1)
		else:
			outs = specs.unsqueeze(1)
		encoder_out = []
		for i in range(len(self.encoder)):
			outs = self.encoder[i](outs)
			#print(f"encoder layer {i} : shape {outs.shape}")
			encoder_out.append(outs)

		B, C, D, T = outs.size()
		outs = outs.view(B, -1, T)
		#print(outs.shape)
		for i in range(len(self.tcmlayer)):
			outs = self.tcmlayer[i](outs)

		outs = outs.view(B, C, D, T)
		#print(outs.shape)
		for i in range(len(self.decoder)):
			outs = self.decoder[i](outs + encoder_out[-1-i])
			#print(f"decoder layer {i} : shape {outs.shape}")
			outs = outs[..., :-1]
		if self.dereverb:
			outs = self.multiframe_filtering(denoise_out, outs)
		else:
			outs = self.multiframe_filtering(inputs, outs)
		return outs

if __name__ == "__main__":
	Net = DN_Net(dereverb=True)
	inputs = torch.randn([10, 161, 1000])
	outs = Net(inputs, inputs)
	print(outs.shape)



