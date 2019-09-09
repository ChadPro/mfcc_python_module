# -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.

import numpy as np
import tensorflow as tf
from os import walk
from os.path import join
import random
import sys
import librosa
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from scipy.fftpack import rfft
from scipy.fftpack import ifft
from scipy import signal
import math

WAV_SOURCE = "test_0.wav"

""" 预加重
"""
def pre_emphasizing(sample, length, factor):
    process_sample = np.zeros(length, dtype=np.float32)
    for i in range(length):
        if i == 0:
            process_sample[i] = sample[i]
        else:
            process_sample[i] = sample[i] - factor * sample[i-1]
    return process_sample

""" 分帧、加窗、fft
"""
def mfccFrame(sample, frameSize, rawDataLen):
    sampleRate = 16000
    hamWinSize = int(sampleRate * 0.025)

    hopStep = int(0.01 * sampleRate)
    frameNum = math.ceil(rawDataLen / hopStep)

    frameSample = np.zeros((frameNum, frameSize), dtype=np.float32)
    fftSample = np.zeros((frameNum, frameSize), dtype=np.float32)

    i = 0
    hamWin = np.hamming(hamWinSize)
    while i*hopStep < rawDataLen:
        for j in range(frameSize):
            if (j < hamWinSize) & (i*hopStep + j < rawDataLen):
                frameSample[i][j] = sample[i*hopStep +j] * hamWin[j]
            else:
                frameSample[i][j] = 0. 
        i += 1
    
    for i,frame in enumerate(frameSample):
        frame_fft = fft(frame)
        frame_fft = [num.real**2+num.imag**2 for num in frame_fft]
        fftSample[i] = np.array(frame_fft, dtype=np.float32)
    
    return fftSample, frameNum

""" Mel频谱
"""
def computeMel(fftSampel, nfilter, frameSize, frameNum):
    sampleRate = 16000
    freMax = sampleRate / 2.0
    freMin = 0.
    melFremax = 1125 * np.log(1 + freMax / 700)
    melFremin = 1125 * np.log(1 + freMin / 700)

    k = (melFremax - melFremin) / (nfilter + 1)
    m = np.zeros(nfilter+2, dtype=np.float32)
    h = np.zeros(nfilter+2, dtype=np.float32)
    f = np.zeros(nfilter+2, dtype=np.float32)

    for i in range(nfilter+2):
        m[i] = melFremin + k * i
        h[i] = 700 * (np.exp(m[i]/1125) - 1)
        f[i] = np.floor((frameSize + 1) * h[i] / sampleRate)
    
    mel = np.zeros((frameNum, nfilter), dtype=np.float32)
    for i in range(frameNum):
        for j in range(1,nfilter+1):
            temp = 0.
            for z in range(frameSize):
                if z < f[j - 1]:
                    temp = 0.
                elif (z >= f[j - 1]) & (z <= f[j]):
                    temp = (z - f[j - 1]) / (f[j] - f[j - 1])
                elif (z >= f[j]) & (z <= f[j + 1]):
                    temp = (f[j + 1] - z) / (f[j + 1] - f[j])
                elif z > f[j + 1]:
                    temp = 0.
                mel[i][j-1] += fftSampel[i][z] * temp
    
    # 取对数
    for i in range(frameNum):
        for j in range(nfilter):
            if (mel[i][j] <= 0.00000000001) | (mel[i][j] >= 0.00000000001):
                mel[i][j] = np.log(mel[i][j])

    return mel

""" DCT
"""
def DCT(mel, frameNum, filterNum):
    c = np.zeros((frameNum, filterNum))
    for k in range(frameNum):
        for i in range(13):
            for j in range(filterNum):
                c[k][i] += mel[k][j] * np.cos(np.pi * i / (2*filterNum) * (2*j+1))
    return c

def wav2mfcc(data, sr, nfilter):
    raw_data_len = len(data)
    # 预加重
    factor = 0.95
    process_sample = pre_emphasizing(data, raw_data_len, factor)
    # 分帧、加窗、fft
    frameSize = 512
    fft, frameNum = mfccFrame(process_sample, frameSize, raw_data_len)
    # Mel频谱
    nfilter = 20
    mel = computeMel(fft, nfilter, frameSize, frameNum)
    # DCT
    c = DCT(mel, frameNum, nfilter)

    return c

def compute_mfcc(file):
    sr, x = wav.read(file)
    x = np.array(x, dtype=np.float32)
    print(len(x))
    mfcc = wav2mfcc(x, sr, 20)

    return mfcc

mfcc_features = compute_mfcc(WAV_SOURCE)
print(mfcc_features.shape)
print(mfcc_features[0])