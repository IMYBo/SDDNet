
"""
some tool func about speech processing 

author: yxhu
"""
'''
读取文件方式修改为soundfile
单通道
'''

import scipy 
import scipy.signal as signal 
import numpy as np
import wave
import soundfile as sf
import scipy.io as sio

def audioread(path, sample_rate=16000, selected_channels=[1]):
    """
        read wave data like matlab's audioread
        selected_channels: for multichannel wave, return selected_channels' data 
    """
    selected_channels = [ x - 1 for x in selected_channels]
    #print(path)
    wavedata,sample_rate = sf.read(path)
    nchannels = 1
    if nchannels == 1:
        return np.reshape(wavedata,[-1])
    else:
        wavedata = np.reshape(wavedata, [nframes,nchannels])
        return wavedata[:, selected_channels]

def audiowrite(path, data, nchannels=1, samplewidth=2, sample_rate=16000):
    
    data = np.reshape(data, [-1, nchannels])
    nframes = data.shape[0]
    fid = wave.open(path, 'wb')
    data *= 32767.0
    fid.setparams((nchannels, samplewidth, sample_rate, nframes,"NONE", "not compressed"))
    fid.writeframes(np.array(data, dtype=np.int16).tostring())
    fid.close()

def enframe(data, window, win_len, inc):
    data_len = data.shape[0] 
    if data_len <= win_len :
        nf = 1
    else:
        nf = int((data_len-win_len+inc)/inc)
    # 2019-3-29:
    # remove the padding, the last points will be discard

    #pad_length = int((nf-1)*inc+win_len)
    #zeros = np.zeros((pad_length - data_len, ))
    #pad_signal = np.concatenate((data, zeros))

    indices = np.tile(np.arange(0,win_len), (nf,1))+ np.tile(np.arange(0,nf*inc, inc), (win_len,1)).T 
    indices = np.array(indices, dtype=np.int32)
    frames = data[indices]
    windows = np.reshape(np.tile(window, nf),[nf,win_len])
    return frames*windows

def fft(data, fft_len):
    return np.abs(np.fft.rfft(data, n=fft_len)) 

def activelev(data,fs):
    """
        Normalize data to 0db
        now is a little func,
        furthermore it will be
        transplantated from matlab version
    """
    max_amp = np.std(data)#np.max(np.abs(data))
    return (data+1e-6)/(max_amp+1e-6)

def resample(src, fs, tfs):
    if fs == tfs:
        return src
    if fs > tfs:
        down_sample(src, fs, tfs)
    else:
        up_sample(src, fs, tfs)

def up_sample(src, fs, tfs):
    """
        up sample
    """
    pass

def down_sample(src, fs, tfs):
    """
        down sample
    """
    pass


if __name__ == '__main__':
    wave_data = audioread('/home//work_nfs2/yxhu/workspace/se-cldnn-torhc/exp/500_with_clip_grad_no_sampler_cldnn_1_1_0.0005_16k_6_9/test_3k/speaker079_10801018.wav').reshape([-1])
    win = np.hamming(400)/1.2607934
    en_data = enframe(wave_data,win,400, 100)
    fft_data=fft(en_data,512)
    sio.savemat('test.mat', {'py_en':en_data, 'py_fft':fft_data, 'py_wave':wave_data})
