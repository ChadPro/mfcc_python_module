from distutils.core import setup, Extension

MOD = "QSAudio"
setup(name = MOD, ext_modules=[Extension(MOD, sources=["mfcc.cpp"])])