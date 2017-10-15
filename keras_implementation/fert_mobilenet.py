#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:47:47 2017

@author: slerp4
"""
# import necessary APIs
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image
from keras import backend as K
import numpy as np
from timeit import default_timer as timer
import os, importlib


# set backend
tf = 'tensorflow'
if K.backend() != tf:
    os.environ['KERAS_BACKEND'] = tf
    importlib.reload(K)
    assert K.backend() == tf
    print('{} backend is sucessfully set'.format(K.backend()))
elif K.backend() == tf:
    print('{} backend has already been set'.format(K.backend()))

# Setup the model
time_load_model_start = timer()
model = MobileNet(input_shape=(224,224,3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling='avg')
time_load_model_end = timer()
print('Deep CNN model InceptionV3 is loaded, time taken: {} seconds'.format(time_load_model_end-time_load_model_start))


# initiate the time record
num_trial = int(input('Please set number of trials: '))
#num_trial = 99
time_elapsed = 0

# create/open a .csv file to record data
file = open('fert_time_record_mobilenet.csv','w+')

for i in range(num_trial):
#    print('This is {}-th trial.\n'.format(i+1))
    time_start = timer()
    # Load input image
    img_path = 'dog.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract feature
    features = model.predict(x)

    time_end = timer()

    # record time used for i-th trial
    time_elapsed += (time_end-time_start)

    # print results on screen
    print('{}-th trial is done! Features of dimension {} are extracted from last pooling layer, using {} seconds.'.format(i+1,features.shape,time_end-time_start))

    # write time record to file
    file.write(str(time_end-time_start)+'\n')

file.close()
#print('\nThe dimension of feature is {}.'.format(features.shape))
print('Average proc. time per image is: {}'.format(time_elapsed/num_trial))
#_,= np.argmax(features)
#print()
