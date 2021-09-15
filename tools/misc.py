
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)
# modified Yanxin Hu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
import os 
import sys

def read_and_config_file(wave_list, decode=0):
    processed_list = []
    
    with open(wave_list) as fid:
        if decode:
            for line in fid:
                tmp = line.strip().split(' ')
                #print(len(tmp))
                if len(tmp)!= 1:
                    sample = {'inputs': tmp[0] + ' ' + tmp[1]}
                else:
                    sample = {'inputs': tmp[0]}
                processed_list.append(sample)

        else:
            for line in fid:
                tmp = line.strip().split(' ')
#               
                #print(tmp[0])
#                print(tmp[1])
#               print(tmp[2])
                if len(tmp) == 2:
                    sample = {'inputs': tmp[0], 'duration':float(tmp[1])}
                elif len(tmp) == 1:
                    sample = {'inputs': tmp[0]}
                else:
                    tmp = line.strip().split('.wav')
                    #print(tmp)
                    tmp[0] += '.wav'
                    #print(tmp[0])
                    #print(len(tmp))
                    sample = {'inputs': tmp[0], 'duration':float(tmp[1].split(' ')[-1])}
                processed_list.append(sample)
    return processed_list

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        try:
            checkpoint = torch.load(checkpoint_path)
        except RuntimeError:
            checkpoint = torch.jit.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def get_learning_rate(optimizer):
    """Get learning rate"""
    return optimizer.param_groups[0]["lr"]


def reload_for_eval(model, checkpoint_dir, use_cuda,load_strict):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        print(checkpoint['model'].keys())
        model.load_state_dict(checkpoint['model'], strict=load_strict)
        print('=> Reload well-trained model {} for decoding.'.format(
            model_name))

def reload_model(model, optimizer, checkpoint_dir, use_cuda=True, strict=True):
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint['model'], strict=strict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        print('=> Reload previous model and optimizer.')
    else:
        print('[!] checkpoint directory is empty. Train a new model ...')
        epoch = 0
        step = 0
    return epoch, step

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}.pt'.format(epoch))
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'step': step}, checkpoint_path)
    with open(os.path.join(checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write('model.ckpt-{}.pt'.format(epoch))
    print("=> Save checkpoint:", checkpoint_path)

def setup_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr
