#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy
import torch 
import random
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as tud
import os 
import sys
sys.path.append(
    os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'speech_processing_toolbox/'))

import voicetool.base as voicebox
import voicetool.utils as utils
import voicetool.multiworkers as worker
from misc import read_and_config_file
import multiprocessing as mp
import random
import librosa
#import scipy as sp
import scipy.signal as sps
import soundfile as sf
import time
"""
去混加上双边滤波
"""
class DataReader(object):
    
    def __init__(self, file_name, win_len=400, win_inc=100,left_context=0,right_context=0, fft_len=512, window_type='hamming', target_mode='Time',sample_rate=16000):
        self.left_context = left_context
        self.right_context = right_context
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = {
                        'hamming':np.hamming(self.win_len)/1.2607934,
                        'hanning':np.hanning(self.win_len),
                        'none':np.ones(self.win_len)
                      }[window_type]
        self.file_list = read_and_config_file(file_name, decode=True)
        self.label_type = target_mode
    
    def extract_feature(self, path):
        path = path['inputs']
        #print(path)
        utt_id = path.split('/')[-1]
        data = voicebox.audioread(path).astype(np.float32)
        if self.label_type == 'Spec':
            inputs = voicebox.enframe(data, self.window, self.win_len,self.win_inc)
            fft_inputs = np.fft.rfft(inputs, n=self.fft_len)
            length, dims = fft_inputs.shape
            inputs = np.reshape(np.abs(fft_inputs), [1, length, dims])
            inputs = inputs.astype(np.float32)
            nsamples = data.shape[0]
            return inputs, np.angle(fft_inputs), utt_id, nsamples
        else:
            inputs = np.reshape(data, [1, data.shape[0]])
            return inputs, utt_id, data.shape[0]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    #print(sr)
    if sr != fs:
        wave_data = librosa.resample(wave_data, sr, fs)
    return wave_data

def get_firstchannel_read(path, fs=16000):
    wave_data, sr = sf.read(path)
    #print(sr)
    if sr != fs:
        wave_data = librosa.resample(wave_data, sr, fs)
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:,0]    
    return wave_data

def activelev(data):###
    # max_val = np.max(np.abs(data))
    eps = np.finfo(np.float32).eps
    max_val = (1. + eps) /( np.std(data) + eps)
    data = data * max_val
    return data
'''
def add_reverb(cln_wav, rir_wav):
    """
    Args:
        :@param cln_wav: the clean wav
        :@param rir_wav: the rir wav
    Return:
        :@param wav_tgt: the reverberant signal
    """
    rir_wav = np.array(rir_wav)
    rir_wav = np.squeeze(rir_wav)
    wav_tgt = sps.oaconvolve(cln_wav, rir_wav)#sp.convolve(cln_wav, rir_wav)
    wav_tgt = np.transpose(wav_tgt)
    return wav_tgt
'''

def add_reverb(cln_wav, rir_wav, return_direct = False):
    """
    Args:
        :@param cln_wav: the clean wav
        :@param rir_wav: the rir wav
    Return:
        :@param wav_tgt: the reverberant signal
    """
    
    rir_wav = np.array(rir_wav)
    rir = rir_wav
    wav_tgt = sps.oaconvolve(cln_wav, rir)
    #### generate dereverb label for training data
    if(return_direct is True):
        rir_direct = np.zeros(rir_wav.shape)
        rir_late = np.zeros(rir_wav.shape)
        clean_len = len(cln_wav)
        #cln_wav = cln_wav[:, np.newaxis]
        predelay_ms = 100
        dt = np.argmax(rir_wav)
        dt = int(dt + (predelay_ms * 16000 / 1000))
        rir_direct[:dt] = rir_wav[:dt]
        rir_late[dt:] = rir_wav[dt:]
        #print(rir_direct.shape)
        #print(cln_wav.shape)
        wav_tgt_direct = sps.oaconvolve(cln_wav, rir_direct, mode='full')[:clean_len]
        wav_tgt_late = sps.oaconvolve(cln_wav, rir_late, mode='full')[:clean_len]
        return wav_tgt, wav_tgt_direct, wav_tgt_late
    else:
        return wav_tgt

def addnoise(clean, noise, rir_path, scale, snr):
    '''
    if rir is not None, the speech of noisy has reverberation
    and return the clean with reverberation
    else no reverberation
    Args:
        :@param clean_path: the path of a clean wav
        :@param noise_path: the path of a noise wav
        :@param start: the start point of the noise wav 
        :@param scale: the scale factor to control volume
        :@param snr:   the snr when add noise
    Return:
        :@param Y: noisy wav
        :@oaram X: clean wav
    '''
    ####八个通道选哪个？

    noise_length = noise.shape[0]
    clean_length = clean.shape[0]
    clean_snr = snr / 2
    noise_snr = -snr / 2
    clean_weight = 10**(clean_snr/20)
    noise_weight = 10**(noise_snr/20)
######
    if clean_length < noise_length: 
        start = np.random.randint(noise_length - clean_length)
        noise_selected = np.zeros(clean_length)
        noise_selected = noise[start : start+clean_length]
        clean_selected = clean
    else:
        noise_selected = noise
        clean_selected = clean
        
    if rir_path is not None:
        rir = audioread(rir_path)
        if len(rir.shape) != 1:
            rir1 = rir[:,0]
            rir2 = rir[:,0]
        else:
            rir1 = rir
            rir2 = rir
        #print(rir1.shape)
        #print(rir2.shape)
        #print('noise_selected', noise_selected.shape)
        #print('clean_selected', clean_selected.shape)
        #print('rir1', rir1.shape)
        #print('rir2', rir2.shape)
        #print(clean_selected.shape)
        rir_clean, clean_direct, clean_late = add_reverb(clean_selected, rir1, True)
        rir_noise, noise_direct, noise_late = add_reverb(noise_selected, rir2, True)
        #print('cnm')
        #return rir_clean, rir_clean, clean_direct, clean_late
    else:
        rir_clean = clean_selected
        rir_noise = noise_selected
        clean_direct = rir_clean
        noise_direct = rir_noise

    rir_noise_n_1 = activelev(rir_noise)
    rir_clean_n_1 = activelev(rir_clean)
    clean_direct = activelev(clean_direct)
    clean_late = activelev(clean_late)
    rir_clean = rir_clean_n_1 * clean_weight
    rir_noise = rir_noise_n_1 * noise_weight
    clean_direct = clean_direct * clean_weight
    clean_late = clean_late * clean_weight
    if(rir_path is not None):
        rir_clean = rir_clean[:len(rir_clean) - len(rir1) + 1]
        rir_noise = rir_noise[:len(rir_noise) - len(rir2) + 1]
        clean_direct = clean_direct[:len(rir_clean)]
        clean_late = clean_late[:len(rir_clean)]
    noisy = rir_clean + rir_noise
    max_amp_0 = np.max(np.abs([rir_noise, rir_clean, noisy, clean_direct, clean_late]))   
    if max_amp_0 == 0:
        max_amp_0 = 1
    mix_scale_0 = 1/max_amp_0*scale
    direct_clean = np.empty((clean_direct.shape))
    late_clean = np.empty((clean_late.shape))
    Y = np.empty((noisy.shape))
    direct_clean = clean_direct * mix_scale_0
    late_clean = clean_late * mix_scale_0
    clean = rir_clean * mix_scale_0
    Y = noisy * mix_scale_0 
    return Y, clean, direct_clean, late_clean

class Processer(object):

    def __init__(self, win_len=400, win_inc=100,left_context=0,right_context=0, fft_len=512, window_type='hamming', target_mode='Time'):
        self.left_context = left_context
        self.right_context = right_context
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = {
                        'hamming':np.hamming(self.win_len)/1.2607934,
                        'none':np.ones(self.win_len)
                      }[window_type]
        self.label_type = target_mode
    
    def process(self, clean_path, noise_path, rir_path, start_time, segement_length, randstat):
        #print('noise :', noise_path)
        #print('clean: ', clean_path)
        #print('rir: ', rir_path)
         
        wave_clean = get_firstchannel_read(clean_path)
        wave_noise = get_firstchannel_read(noise_path)

        if start_time == -1:
            wave_clean = np.concatenate([wave_clean, wave_clean[:segement_length-wave_clean.shape[0]]])
        else:
            wave_clean = wave_clean[start_time:start_time+segement_length]
         
        # I find some sample are not fixed to segement_length,
        # so i padding zero it into segement_length
        if wave_clean.shape[0] != segement_length:
            padded_inputs = np.zeros(segement_length, dtype=np.float32)
            padded_inputs[:wave_clean.shape[0]] = wave_clean
        else:
            padded_inputs = wave_clean

        if wave_noise.shape[0] < segement_length:
            padded_noise = np.zeros(segement_length, dtype=np.float32)
            padded_noise[:wave_noise.shape[0]] = wave_noise
            wave_noise = padded_noise

        method = randstat.random()
        scale = randstat.uniform(0.3, 0.9)
        snr = randstat.uniform(-5, 20)

        inputs, labels_full, labels_direct, labels_late = addnoise(padded_inputs, wave_noise, rir_path, scale, snr)

        return inputs, labels_full, labels_direct, labels_late

class TFDataset(Dataset):

    def __init__(self,
                 clean_scp,
                 noise_scp,
                 rir_scp,
                 processer,
                 repeat,
                 segement_length=4,
                 sample_rate=16000,
                 gender2spk=None
                ):
        '''
            scp_file_name: the list include:[input_wave_path, output_wave_path, duration]
            spk_emb_scp: a speaker embedding ark's scp 
            segement_length: to clip data in a fix length segment, default: 4s
            sample_rate: the sample rate of wav, default: 16000
            processer: a processer class to handle wave data 
            gender2spk: a list include gender2spk, default: None
        '''
        self.clean_list = read_and_config_file(clean_scp)
        self.noise_list = read_and_config_file(noise_scp)
        self.rir_list = read_and_config_file(rir_scp)
        self.repeat = repeat
        self.processer = processer
        mgr = mp.Manager()
        self.index =mgr.list()#[d for b in buckets for d in b]
        self.index *= repeat
        self.segement_length = segement_length * sample_rate
        self._dochunk(SAMPLE_RATE=sample_rate)
        self.randstats = [ np.random.RandomState(idx) for idx in range(3000)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_info, start_time = self.index[index]
        len_noise = len(self.noise_list)
        len_rir = len(self.rir_list)
        randstat = self.randstats[(index + 11) % 3000]
        cnt_noise = randstat.randint(0, len_noise-1)
        cnt_rir = randstat.randint(0, len_rir-1)
        inputs, labels_full, labels_direct, labels_late = self.processer.process(data_info['inputs'], self.noise_list[cnt_noise]['inputs'], self.rir_list[cnt_rir]['inputs'],start_time, self.segement_length, randstat)
        return inputs, labels_full, labels_direct, labels_late

    def _dochunk(self, SAMPLE_RATE=16000, num_threads=12):
        # mutliproccesing
        def worker(target_list, result_list, start, end, segement_length, SAMPLE_RATE):
            for item in target_list[start:end]:
                duration = item['duration']
                length = duration*SAMPLE_RATE
                if length < segement_length:
                    if length * 2 < segement_length:
                        continue
                    result_list.append([item, -1])
                else:
                    sample_index = 0
                    while sample_index + segement_length < length:
                        result_list.append(
                            [item, sample_index])
                        sample_index += segement_length
                    if sample_index != length - 1:
                        result_list.append([
                            item,
                            int(length - segement_length),
                        ])
        pc_list = []
        stride = len(self.clean_list) // num_threads
        if stride < 100:
            p = mp.Process(
                            target=worker,
                            args=(
                                    self.clean_list,
                                    self.index,
                                    0,
                                    len(self.clean_list),
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                        )
            p.start()
            pc_list.append(p)
        else: 
            for idx in range(num_threads):
                if idx == num_threads-1:
                    end = len(self.clean_list)
                else:
                    end = (idx+1)*stride
                p = mp.Process(
                                target=worker,
                                args=(
                                    self.clean_list,
                                    self.index,
                                    idx*stride,
                                    end,
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                            )
                p.start()
                pc_list.append(p)
        for p in pc_list:
            p.join()
            p.terminate()

class Sampler(tud.sampler.Sampler):
    '''
     
    '''
    def __init__(self, data_source, batch_size):
        #print(len(data_source))
        #print(batch_size)
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i+batch_size)
                        for i in range(0, it_end, batch_size)]
        self.data_source = data_source
        
    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = np.array([len(inputs), max_t, inputs[0].shape[1]])
    inputs_mat = np.zeros(shape, np.float32)
    for idx, inp in enumerate(inputs):
        inputs_mat[idx, :inp.shape[0],:] = inp
    return inputs_mat

def collate_fn(data):
    inputs, s, s1, s2 = zip(*data) 
    #print
    #cnt = 0
    #for a in inputs:
    #    print(cnt, ' ', a.shape)
    #    cnt = cnt + 1 
    #print(inputs.shape)
    inputs = np.array(inputs, dtype=np.float32)
    s = np.array(s, dtype=np.float32)
    s1 = np.array(s1, dtype=np.float32)
    s2 = np.array(s2, dtype=np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(s), torch.from_numpy(s1), torch.from_numpy(s2)

def make_loader(clean_scp, noise_scp, rir_scp=None, batch_size=8, repeat=2, num_workers=12, segement_length = 8, sample_rate=16000, processer=Processer()):
    dataset = TFDataset( 
                         clean_scp=clean_scp,
                         noise_scp=noise_scp,
                         rir_scp=rir_scp,
                         repeat=repeat,
                         segement_length=segement_length,
                         sample_rate=sample_rate,
                         processer=processer
                        )
    sampler = Sampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            #sampler=sampler, 
                            drop_last=False,
                            shuffle=True
                        )
                            #shuffle=True,
    return loader, None #, Dataset
if __name__ == '__main__':
    laoder,_ = make_loader(clean_scp = '../data/clean_cv.lst',#'/home/work_nfs3/lvshubo/dasan/lstm/sruc_16k_auto_noise/data/clean_tr.lst', 
                           noise_scp = '../data/noise_cv.lst',#'/home/work_nfs3/lvshubo/dasan/lstm/sruc_16k_auto_noise/data/noise_tr.lst', 
                           rir_scp = '../data/rir_cv.lst',#'/home/work_nfs3/lvshubo/dasan/lstm/sruc_16k_auto_noise/data/rir_tr.lst', 
                           batch_size = 1, 
                           repeat = 2, 
                           num_workers=16)
                       
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import time 
    import soundfile as sf
    stime = time.time()
    cnt = 0
    os.mkdir('test_wav/input')
    os.mkdir('test_wav/label')
    os.mkdir('test_wav/label_early')
    os.mkdir('test_wav/label_lately')
    for epoch in range(10):
        for idx, data in enumerate(laoder):
            print(len(data))
            inputs, labels, labels1, labels2= data 
            sf.write('test_wav/input/inputs_' + str(cnt) + '.wav', inputs[0], 16000)
            sf.write('test_wav/label/inputs_' + str(cnt) + '.wav', labels[0], 16000)
            sf.write('test_wav/label_early/inputs_' + str(cnt) + '.wav', labels1[0], 16000)
            sf.write('test_wav/label_lately/inputs_' + str(cnt) + '.wav', labels2[0], 16000)
            cnt = cnt + 1
            #inputs.cuda()
            #labels.cuda()
            if idx%100 == 0:
                etime = time.time()
                print(epoch, idx, labels1.size(), (etime-stime)/100)
                stime = etime
