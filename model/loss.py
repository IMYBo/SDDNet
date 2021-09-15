#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn.functional as F 


def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data

def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)

def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)

def si_snr(estimated, original,EPS=1e-8):
    # estimated = remove_dc(estimated)
    # original = remove_dc(original)
    target = pow_norm(estimated, original) * original / (pow_p_norm(original) + EPS)
    noise = estimated - target
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return torch.mean(sdr)

def sd_snr(estimated, original, EPS=1e-8):
    target = pow_norm(estimated, original) * original / (pow_p_norm(original) + EPS)
    noise = estimated - original
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return torch.mean(sdr)

def GD(inputs, labels):
    gth_real, gth_imag = torch.chunk(inputs, 2, 1)
    est_real, est_imag = torch.chunk(labels, 2, 1) 
    
    gth_phase = torch.atan2(gth_imag, gth_real+1e-8)
    est_phase = torch.atan2(est_imag, est_real+1e-8)
    
    gth_gd = gth_phase[:,1:] - gth_phase[:,:-1]
    est_gd = est_phase[:,1:] - est_phase[:,:-1]
    
    gth_gd = torch.cos(gth_gd)
    est_gd = torch.cos(est_gd)

    return F.mse_loss(est_gd, gth_gd)

def kl(estimated, labels): 
    est_real, est_imag = torch.chunk(estimated, 2, 1)
    gth_real, gth_imag = torch.chunk(labels, 2, 1)
    
    est_mag = torch.sqrt(est_real**2+est_imag**2+1e-8)     
    gth_mag = torch.sqrt(gth_real**2+gth_imag**2+1e-8)     

    loss = est_mag*torch.log(est_mag/(gth_mag+1e-12)+1e-12)# - (est_mag - est_mag)
    return torch.mean(loss)

def mix(inputs, labels,weight=[1,1,1]): 
    est_spec, est_wav = inputs 
    gth_spec, gth_wav = labels 
    #nse_wav, nse_spec = gth_wav-est_wav, gth_spec - est_spec
    loss1 = -si_snr(est_wav, gth_wav)
    loss2 = F.mse_loss(est_spec, gth_spec)
    #loss3 = rgkl(est_spec, gth_spec)
    loss = loss1*weight[0]+loss2*weight[1]#+loss3*weight[2]
    return loss

def delta(inputs):
    return inputs[...,1:]-inputs[...,:-1]

def mix_loss(inputs,labels):
    return inputs[...,1:]-labels[...,:-1]

def gd(inputs):
    return inputs[:,1:]-inputs[:,:-1]
