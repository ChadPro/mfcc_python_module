import QSAudio
import numpy as np

""" Test one time
"""
mfcc_features = QSAudio.mfcc("test/test_0.wav", 20)
mfcc_features = np.array(mfcc_features, dtype=np.float32)
mfcc_features = np.reshape(mfcc_features, (-1, 20))
print(mfcc_features.shape)
print(mfcc_features[0])

""" Test 10000 times
"""
# for i in range(10000):
#     mfcc_features = QSAudio.mfcc("test/test_0.wav", 20)
#     mfcc_features = np.array(mfcc_features, dtype=np.float32)
#     mfcc_features = np.reshape(mfcc_features, (-1, 20))
#     if i % 1000 == 0:
#         print(i)
