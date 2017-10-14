#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:47:47 2017

@author: slerp4
"""
# import necessary APIs
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
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
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# initiate the time record
num_trial = int(input('Please set number of trials: '))
time_elapsed = 0

# create/open a .csv file to record data
file = open('fert_time_record_vgg16.csv','w+')

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
    print('{}-th trial is done! Features from block 5 are extracted, using {} seconds.'.format(i+1,time_end-time_start))
    
    # write time record to file
    file.write(str(time_end-time_start)+'\n')

file.close()
#print('\nThe dimension of feature is {}.'.format(features.shape))
print('Average proc. time per image is: {}'.format(time_elapsed/num_trial))
#_,= np.argmax(features)
#print()
