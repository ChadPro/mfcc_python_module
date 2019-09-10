# -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.
import librosa
import sys
import scipy.io.wavfile as wav
import numpy as np

WAV_SOURCE = "test/test_1.wav"

def main(argv):
    cmd = 0
    if len(argv) > 1:
        cmd = int(argv[1])
    
    if cmd == 0:
        sr, wavsignal = wav.read(WAV_SOURCE)
        x = np.array(wavsignal, dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
        mfcc_features = mfcc.T
        print(mfcc_features.shape)
        print(mfcc_features[0])

    elif cmd == 1:
        sr, wavsignal = wav.read(WAV_SOURCE)
        x = np.array(wavsignal, dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_fft=512, hop_length=160, win_length=400, window="hamming", center=False, n_mels=20)
        mfcc_features = mfcc.T
        print(mfcc_features.shape)
        print(mfcc_features[0])

if __name__ == "__main__":
    main(sys.argv)