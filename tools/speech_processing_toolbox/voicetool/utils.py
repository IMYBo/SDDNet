import sys
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import struct
def read_binary_file(filename, offset=0):
    """Read data from matlab binary file (row, col and matrix).

    Returns:
        A numpy matrix containing data of the given binary file.
    """
    read_buffer = open(filename, 'rb')
    read_buffer.seek(int(offset), 0)
    header = struct.unpack('<xcccc', read_buffer.read(5))
    if header[0] != 'B':
        print("Input .ark file is not binary")
        sys.exit(-1)
    if header[1] == 'C':
        print("Input .ark file is compressed, exist now.")
        sys.exit(-1)

    rows = 0; cols= 0
    _, rows = struct.unpack('<bi', read_buffer.read(5))
    _, cols = struct.unpack('<bi', read_buffer.read(5))

    if header[1] == "F":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 4),
                                dtype=np.float32)
    elif header[1] == "D":
        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 8),
                                dtype=np.float64)
    mat = np.reshape(tmp_mat, (rows, cols))

    read_buffer.close()

    return mat


def pcen(data):
    data = np.exp(data)
    M = signal.lfilter([0.025], [1, -0.975], data)
    smooth = (1e-6 + M)**0.98
    data = (data / smooth + 2)**0.5 - 2**0.5
    return data


def utt_cmvn(data):
    len, dim = data.shape
    slide_mean = np.mean(data,axis=0)
    #slide_std = np.std(data,axis=0)
    result = (data - slide_mean)
    return result


def window_cmvn(data):
    sliding_window = 100
    len, dim = data.shape
    result = np.zeros((len, dim))
    for i in range(0, len):
        start = np.max([0, i - sliding_window])
        slide_mean = np.mean(data[start:i],axis=0)
    #    slide_std = np.std(data[start:i],axis=0)
        result[i] = (data[i] - slide_mean)
    return result


def splice_feats(data, left=0, right=0):
    length, dims = data.shape
    sfeats = []
    if left != 0:
    # left 
        for i in range(left, 0, -1):
            t = data[:length-i]
            for j in range(i):
                t = np.pad(t, ((1, 0), (0, 0)), 'symmetric')
            sfeats.append(t)
    sfeats.append(data)
    # right
    if right != 0:
        for i in range(1,right+1):
            t = data[i:]
            for j in range(i):
                t = np.pad(t, ((0, 1 ), (0, 0)), 'symmetric')
            sfeats.append(t)
    return np.concatenate(np.array(sfeats), 1)

def test():
    a = np.arange(15).reshape([-1,3])
    b = splice_feats(a,left= 0,right=0)
    print(b.shape)
test()
