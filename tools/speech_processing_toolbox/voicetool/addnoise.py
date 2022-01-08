import sys
import os

import scipy.io as sio
import scipy
import numpy as np
import multiprocessing
import wave

def activelev(data):
    '''
        need to update like matlab
    '''
    max_amp = np.max(np.abs(data))
    return data/max_amp

def add_noisem(clean_path, noise_path, out_clean_dir, out_noisy_dir, start, scale, snr):
    clean = read(clean_path)
    noise = read(noise_path)
    
    cname = clean_path.split('/')[-1].split('.wav')[0]
    nname = noise_path.split('/')[-1].split('.wav')[0]
    name = cname+'_'+str(snr)+'_'+nname+'_'+str(-snr)+'.wav'
    clean_size = clean.shape[0]
    noise_selected = noise[start:start+clean_size]
    clean_n = activelev(clean)
    noise_n = activelev(noise_selected)
    clean_snr = snr
    noise_snr = -snr
    clean_weight = 10**(clean_snr/20)
    noise_weight = 10**(noise_snr/20)
    clean = clean_n * clean_weight
    noise = noise_n * noise_weight
    noisy = clean + noise
    max_amp = np.max(np.abs([noise, clean, noisy]))
    mix_scale = 1/max_amp*scale
    X = clean * mix_scale
    # N = noise * mix_scale
    Y = noisy * mix_scale
    
    write(out_clean_dir+'/'+name, X)
    write(out_noisy_dir+'/'+name, Y)


def read(path):
    """
        read wave data like matlab's audioread
    """
    fid = wave.open(path, 'rb')
    params = fid.getparams()
    nchannels, samplewidth, framerate, nframes = params[:4]
    strdata = fid.readframes(nframes*nchannels)
    fid.close()
    wavedata = np.fromstring(strdata, dtype=np.int16)
    wavedata = wavedata*1.0/(32767.0*samplewidth/2)
    return np.reshape(wavedata, [-1])

def write(path, data):

    nchannels = 1
    samplewidth=2
    sample_rate=16000
    data = np.reshape(data, [-1, 1])
    nframes = data.shape[0]
    fid = wave.open(path, 'wb')
    data *= 32767.0
    fid.setparams((nchannels, samplewidth, sample_rate, nframes, "NONE", "not compressed"))
    fid.writeframes(np.array(data, dtype=np.int16).tostring())
    fid.close()

def AddNoise(mix_list, out_clean_dir, out_noisy_dir, num_threads=12):
    pool = multiprocessing.Pool(num_threads)
    if not os.path.isdir(out_clean_dir):
        os.mkdir(out_clean_dir)
    if not os.path.isdir(out_noisy_dir):
        os.mkdir(out_noisy_dir)
    with open(mix_list) as fid:
        for line in fid:
            tmp = line.strip().split()
            cname, nname, start, snr, scale = tmp
            start = int(start)
            scale = float(scale)
            snr = float(snr)
            pool.apply_async(
                add_noisem, args=(
                            cname,
                            nname,
                            out_clean_dir,
                            out_noisy_dir, 
                            start,
                            scale,
                            snr,)
            )
            
    pool.close()
    pool.join()
    

def generate_mix_list(cwav_list, nwav_list, output_list, snr_range=[-5,5]):
    '''
        cwav_list: include clean wav path list
        nwav_list: include noise wav path list
        output_list: output cwav path, nwav_path, start_time, scale snr
    '''
    noise_lists = []
    
    with open(nwav_list) as nfid:
        for line in nfid:
            noise_lists.append(line.strip())
    
    noise_lists_length = len(noise_lists)
    with open(cwav_list) as cfid:
        with open(output_list, 'w') as outfid:
            for line in cfid:
                cpath = line.strip()
                cwav_len = read(cpath).shape[0]
                while True:
                    nid = np.random.randint(noise_lists_length)
                    nwav_len = read(noise_lists[nid]).shape[0]
                    if nwav_len < cwav_len:
                        continue
                    else:
                        break
                if cwav_len < nwav_len:
                    stime = np.random.randint(nwav_len-cwav_len)
                else:
                    stime = 0
                if isinstance(snr_range, list):
                    snr = (snr_range[1]-snr_range[0])*np.random.ranf()+snr_range[0]
                else:
                    snr = snr_range
                #while True:
                    # the vol scale is N(0.9,0.2)
                 #   scale = np.random.randn()*.2 + 0.9
                 #   if scale <= 1.0 and scale > 0:
                 #       break;
                scale = 0.9
                outfid.writelines(cpath+' '+ noise_lists[nid] +' ' + str(stime)+' {:.3f}'.format(snr)+' {:.3f}'.format(scale)+'\n')
                sys.stdout.flush()

if __name__ == '__main__':
    test_clean = './clean_testset.lst'
    test_noise = './noise_testset'
    name = sys.argv[1]
    snr = int(sys.argv[2])
    generate_mix_list(test_clean, test_noise, name, snr_range=snr)
    print('generated mix list')
    AddNoise(name, 'out_clean_'+str(snr), 'out_noisy_'+str(snr), num_threads=12)


