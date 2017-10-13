#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:15:33 2017

@author: slerp4
"""

from timeit import default_timer as timer
from my_lib.backend_switch import set_backend # most time-consuming operation ~7s
from my_lib.vgg16_feature_extraction import vgg16_feature_extractor as extract

# set backend
backend_selection = input('\nSelect keras backend:\n[t]ensorflow(default), thean[o]\t')
if backend_selection == 't':
    set_backend('tensorflow')
elif backend_selection == 'o':
    set_backend('theano')
else:
    set_backend('tensorflow')

# run feature extractor
time_start=timer()
feature = extract(img_path='dog.jpg')
time_end=timer()
print('\nTime taken for vgg16_feature_extractor is: {} seconds'.format(time_end-time_start))
