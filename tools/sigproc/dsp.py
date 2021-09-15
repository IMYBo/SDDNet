#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  ASLP@NPU    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile

sys.path.append(os.path.dirname(sys.path[0]))
from mtxing_misc.logger import logger
log = logger(__file__, 'debug')


MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps
MAX_EXP = np.log(np.finfo(np.float32).max - 10.0)


def wavread(path):
    #wav, sample_rate = sf.read(path, dtype='float32')
    wav, sample_rate = sf.read(path)
    return wav, sample_rate


def wavwrite(signal, sample_rate, path):
    signal = (signal * MAX_INT16).astype(np.int16)
    wavfile.write(path, sample_rate, signal)


def get_window(window_size, window_type, square_root_window=True):
    """Return the window"""
    window = {
        'hamming': np.hamming(window_size),
        'hanning': np.hanning(window_size),
    }[window_type]
    if square_root_window:
        window = np.sqrt(window)
    return window


def fft_point(dim):
    assert dim > 0
    num = math.log(dim, 2)
    num_point = 2**(math.ceil(num))
    return num_point


def pre_emphasis(signal, coefficient=0.97):
    """Pre-emphasis original signal
    y(n) = x(n) - a*x(n-1)
    """
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def de_emphasis(signal, coefficient=0.97):
    """De-emphasis original signal
    y(n) = x(n) + a*x(n-1)
    """
    length = signal.shape[0]
    for i in range(1, length):
        signal[i] = signal[i] + coefficient * signal[i - 1]
    return signal


def stft(signal,
         sample_rate,
         frame_length=32,
         frame_shift=8,
         window_type="hanning",
         preemphasis=0.0,
         square_root_window=True):
    """Compute the Short Time Fourier Transform.

    Args:
        signal: input speech signal
        sample_rate: waveform data sample frequency (Hz)
        frame_length: frame length in milliseconds
        frame_shift: frame shift in milliseconds
        window_type: type of window
        square_root_window: square root window
    Return:
        fft: (n/2)+1 dim complex STFT restults
    """
    if preemphasis != 0.0:
        signal = pre_emphasis(signal, preemphasis)
    hop_length = int(sample_rate * frame_shift / 1000)
    win_length = int(sample_rate * frame_length / 1000)
    num_point = fft_point(win_length)
    window = get_window(num_point, window_type, square_root_window)
    feat = librosa.stft(signal, n_fft=num_point, hop_length=hop_length,
                        win_length=win_length, window=window)
    return np.transpose(feat)


def get_phase(signal,
              sample_rate,
              frame_length=32,
              frame_shift=8,
              window_type="hanning",
              preemphasis=0.0,
              square_root_window=True):
    """Compute phase imformation.

    Args:
        signal: input speech signal
        sample_rate: waveform data sample frequency (Hz)
        frame_length: frame length in milliseconds
        frame_shift: frame shift in milliseconds
        window_type: type of window
        square_root_window: square root window
    """
    feat = stft(signal, sample_rate, frame_length, frame_shift,
                window_type, preemphasis, square_root_window)
    phase = np.angle(feat)
    return phase


def overlap_and_add(spectrum,
                    signal,
                    sample_rate,
                    frame_length=32,
                    frame_shift=8,
                    window_type="hanning",
                    preemphasis=0.0,
                    use_log=False,
                    use_power=False,
                    square_root_window=True):
    """Convert frames to signal using overlap-and-add systhesis.
    Args:
        spectrum: magnitude spectrum
        signal: wave signal to supply phase information
    Return:
        wav: synthesied output waveform
    """
    if use_log:
        spectrum = np.clip(spectrum, a_min=None, a_max=MAX_EXP)
        spectrum = np.exp(spectrum)
    if use_power:
        spectrum = np.sqrt(spectrum)
    phase = get_phase(signal, sample_rate, frame_length, frame_shift,
                      window_type, preemphasis, square_root_window)
    spectrum = spectrum * np.exp(1.0j * phase)
    if spectrum.shape != phase.shape:
        log.show(('Wave and Spectrum are not the same length, '
                     'phase.shape = {}, spectrum.shape = {}').format(
                         spectrum.shape, phase.shape), 'error')
    spectrum = np.transpose(spectrum)
    hop_length = int(sample_rate * frame_shift / 1000)
    win_length = int(sample_rate * frame_length / 1000)
    num_point = fft_point(win_length)
    window = get_window(num_point, window_type, square_root_window)
    wav = librosa.istft(spectrum, hop_length=hop_length,
                        win_length=num_point, window=window)
    if preemphasis != 0.0:
        wav = de_emphasis(wav, preemphasis)
    return wav
