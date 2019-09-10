# -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.
import QSAudio
import sys
import numpy as np

WAV_SOURCE = "test/test_1.wav"

def main(argv):
    cmd = 0
    if len(argv) > 1:
        cmd = int(argv[1])
    
    if cmd == 0:
        """ Test one time
        """
        mfcc_features = QSAudio.mfcc(WAV_SOURCE, 20)
        mfcc_features = np.array(mfcc_features, dtype=np.float32)
        mfcc_features = np.reshape(mfcc_features, (-1, 20))
        print(mfcc_features.shape)
        print(mfcc_features[0])
        print(mfcc_features[int(mfcc_features.shape[0]/2)])
        print(mfcc_features[-1])

    elif cmd == 1:
        """ Test 10000 times
        """
        for i in range(10000):
            mfcc_features = QSAudio.mfcc(WAV_SOURCE, 20)
            mfcc_features = np.array(mfcc_features, dtype=np.float32)
            mfcc_features = np.reshape(mfcc_features, (-1, 20))
            if i % 1000 == 0:
                print(i)

if __name__ == "__main__":
    main(sys.argv)
