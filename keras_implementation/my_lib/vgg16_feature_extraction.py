#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:54:07 2017

@author: slerp4
"""
# import keras deep learning essentials
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np

# import other api
from timeit import default_timer as timer

def vgg16_feature_extractor(img_path='dog.jpg'):

    # load complete VGG-16 cnn model with weights trained on ImageNet
    base_model = VGG16(weights='imagenet')

    # the model below is VGG-16 with last fully-connected layer 'fc2' excluded
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    time_start = timer()
    # Load input image and proprocessing
    # img_path = 'dog.jpg' is default test image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract feature, get the output by method .predict()
    features = model.predict(x)

    time_end = timer()
    # Print results
    print('\nDone! Features of dimension {} are extracted from 2nd last FC layer.'.format(features.shape))
    print('\nTime elapsed: {}'.format(time_end-time_start))

    return features
