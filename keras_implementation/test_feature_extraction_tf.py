#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:15:33 2017

@author: slerp4
"""

from timeit import default_timer as timer
from my_lib.vgg16_feature_extraction import vgg16_feature_extractor as extract
from keras import backend as K
import os
import importlib

# set backend
tf = 'tensorflow'
if K.backend() != tf:
    os.environ['KERAS_BACKEND'] = tf
    importlib.reload(K)
    assert K.backend() == tf
    print('{} backend is sucessfully set'.format(K.backend()))
elif K.backend() == tf:
    print('{} backend has already been set'.format(K.backend()))


# run feature extractor
time_start=timer()
feature = extract(img_path='dog.jpg')
time_end=timer()
print('\nTime taken for vgg16_feature_extractor is: {} seconds'.format(time_end-time_start))
