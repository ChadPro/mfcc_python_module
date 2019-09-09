# mfcc_python_module
## 1. Install
编辑**mfcc.cpp**第３行
```
#include "/home/**/anaconda2/envs/learn/include/python3.5m/Python.h"
```
修改为你电脑中对应的路径，后
```
python setup.py install
```

## 2. Use
在python中的接口使用方法为:
```
import QSAudio

mfcc_features = QSAudio.mfcc("test/test_0.wav", 20)
```
第一个参数是wav文件地址，第二个参数是filters数量

## 3. 说明
资源中，mfcc_ori.py为纯python编写的相同功能代码，可以用来做对比