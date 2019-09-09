# -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.
from distutils.core import setup, Extension

MOD = "QSAudio"
setup(name = MOD, ext_modules=[Extension(MOD, sources=["mfcc.cpp"])])