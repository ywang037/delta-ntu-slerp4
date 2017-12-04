#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:47:47 2017

@author: slerp4
"""

import numpy as np
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.optimizers import SGD
from keras import backend as K
import os, importlib


# set tensorflow as backend 
tf = 'tensorflow'
if K.backend() != tf:
    os.environ['KERAS_BACKEND'] = tf
    importlib.reload(K)
    assert K.backend() == tf
    print('{} backend is sucessfully set'.format(K.backend()))
elif K.backend() == tf:
    print('{} backend has already been set'.format(K.backend()))

'''
#use the codes of these lines to monitor time taken by certian operation
from timeit import default_timer as timer
time_load_model_start = timer()
time_load_model_end = timer()
print('Deep CNN model Mobil9eNet is loaded, time taken: {} seconds'.format(time_load_model_end-time_load_model_start))
'''

# Setup the model    
# the fully connected layer is included for training purpose
# using CASIA-WebFaces dataset for training, 10575 identities in total
# using portion of the entire dataset, 100 identities in trial version 
model = MobileNet(alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights=None, input_tensor=None, pooling=None, classes=100)
#model.summary()

# config the optimizer - stochasitic gradient descent is used here
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# compile and train the model
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])

history = alexnet.fit_generator(train_generator,
                        samples_per_epoch=2000,
                        validation_data=validation_generator,
                        nb_val_samples=800,
                        nb_epoch=80,
                        verbose=1)

#plot_performance(history)

'''
# the above is equivalent to the following:

# compile and train the model
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
print('Test loss:'.format(score[0]))
print('Test accuracy'.format(score[1]))
'''


'''
# the score can also be displayed as
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print('Test loss:'.format(loss))
print('Test accuracy'.format(accuracy))
'''

