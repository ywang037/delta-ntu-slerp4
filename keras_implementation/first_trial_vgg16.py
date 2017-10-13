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
import numpy as np
from timeit import default_timer as timer

# import self-created backend switch API 
from backend_switch import set_keras_backend as set_backend
'''
# Specify the backend
backend_option = input('Please specify backend: ')
set_backend('backend_option')
'''
# Specify the backend
set_backend('tensorflow')

# Setup the model
#model = VGG16(weights='imagenet', include_top=False)

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

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
# Print results
print('\nDone! Features from block 5 are extracted.')
print('\nThe dimension of feature is {}.'.format(features.shape))
print('\nTime elapsed: {}'.format(time_end-time_start))
#_,= np.argmax(features)
#print()
