#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:47:47 2017

@author: slerp4

Compared with _debug version, this version excludes RMSprop optimizer
"""

#import tensorflow as tf
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, CSVLogger
import os, importlib
from timeit import default_timer as timer
import datetime
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf

# check and set tensorflow as backend 
if K.backend() != 'tensorflow':
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    importlib.reload(K)
    assert K.backend() == 'tensorflow'
    print('{} backend is sucessfully set'.format(K.backend()))
elif K.backend() == 'tensorflow':
    print('{} backend has already been set'.format(K.backend()))


# force to use gpu:0 tesla k20c
# Creates a graph.
with tf.device('/device:GPU:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

# training hyper parameters
train_data_dir = '.\Datasets\casia-1771'
numclass = 1771
num_train_samples = 233505
batch_size = 64 
#epochs = 100
alpha = 0.5 # choices=[0.25, 0.5, 0.75, 1.0]
inputsize = 224 # choices=[128, 160, 192, 224, 224], >=32 is ok

'''
# define step decay function - used to visualize learning rate change
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('Current learning rate:', step_decay(len(self.losses)))
'''
        
# learning rate schedule
def step_decay(epoch):
	# initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 20.0
	lrate = init_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


# Setup the model    
# using CASIA-WebFaces dataset for training, 10575 identities in total
model = MobileNet(alpha=alpha, depth_multiplier=1, dropout=1e-3, 
                  include_top=True, weights=None, input_tensor=None, pooling=None, classes=numclass)
model.summary()
print('\nPrepare to train cnn model {}-MobileNet-224 with top layer included'.format(alpha))
#print('Total classes: {}'.format(numclass))
#print('Training samples: {}'.format(num_train_samples))

optimizer_chosen = input('Optimizer (A: SGD/B: Adam)? ')
while optimizer_chosen not in ['A', 'B']:
    optimizer_chosen = input('Optimizer (A: SGD/B: Adam)? ')

epochs = int(input('Number of epochs? '))
while epochs < 0:
    epochs = int(input('Use a positive integer as the number of epochs: '))

init_lr = float(input('Initial learning rate? '))
while init_lr < 0 or init_lr>0.2:
    init_lr = float(input('Use a learning rate in [0, 0.2]: '))

# preparing training data
print('\nDataset path: '+ train_data_dir)
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# load training and testing data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size)

# define the format of names of several files
stamp = str(alpha)+'-mobilenet-'+str(inputsize)+'-c{}-'.format(numclass)+'b{}-'.format(batch_size)+'e{}-'.format(epochs)

if optimizer_chosen == 'A':
    # using step-decaying sgd
    method = 'SGD'
    print('\nUsing step-decaying stochastic gradient descent')
    print('learning rate folds every 20 epochs')
    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    # compile the model
    # loss = mse can be tried also
    train_start = timer()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    '''
    # use following scripts to have learning rate displayed
    # learning schedule callback
    loss_history = LossHistory() 
    lrate = LearningRateScheduler(step_decay)
    # training logger callback, log in csv file
    record = stamp + method
    csv_logger = CSVLogger(record+'.csv',append=True, separator=',')
    callbacks_list = [loss_history, lrate, csv_logger]
    # train the model
    history = model.fit_generator(train_generator, steps_per_epoch=num_train_samples//batch_size, 
                                  epochs=epochs, validation_data=None, callbacks=callbacks_list, verbose=2)
    '''
    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)
    # training logger callback, log in csv file
    record = stamp + method + '-lr{}'.format(init_lr)
    csv_logger = CSVLogger(record+'.csv',append=True, separator=',')
    callbacks_list = [lrate, csv_logger]
    # train the model
    history = model.fit_generator(train_generator, steps_per_epoch=num_train_samples//batch_size, 
                                  epochs=epochs, validation_data=None, callbacks=callbacks_list, verbose=1)
elif optimizer_chosen == 'B':
    # using adam update as adaptive learning rate method
    method = 'Adam'
    print('\nUsing using adam update as adaptive learning rate method')
    adam = Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # original lr=0.001
    # compile the model
    # loss = mse can be tried also
    train_start = timer()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # training logger callback, log in csv file
    record = stamp + method + '-lr{}'.format(init_lr)
    csv_logger = CSVLogger(record+'.csv',append=True, separator=',')
    # train the model
    history = model.fit_generator(train_generator, steps_per_epoch=num_train_samples // batch_size,
                              epochs=epochs, validation_data=None, callbacks=[csv_logger], verbose=1)

    
train_end = timer()
mins, secs = divmod(train_end-train_start,60)
hour, mins = divmod(mins,60)
print('Training process took %d:%02d:%02d' % (hour,mins,secs))

# set a stamp of file name for saving the record and weights
now = datetime.datetime.now() #current date and time
save_name = record +'-'+now.strftime("%Y%m%d-%H%M")

#print(history.history)
print(history.history.keys())
# print plots of acc and loss in one pdf
pp = PdfPages(save_name +'.pdf')
# summarize history for accuracy
plt.plot(history.history['acc']) # plt.plot(history.history['val_acc'])
plt_title = str(alpha)+'-mobilenet-'+str(inputsize)+' trained on small dataset'
plt_legend = method + ', {} classes'.format(numclass)+', batch size ={}'.format(batch_size)
plt.title(plt_title)
plt.ylabel('Model accuracy')
plt.xlabel('Epoch')
plt.legend([plt_legend], loc='lower right')
pp.savefig()
plt.show()
# summarize history for loss
plt.plot(history.history['loss']) #plt.plot(history.history['val_loss'])
plt.title(plt_title)
plt.ylabel('Model loss')
plt.xlabel('Epoch')
plt.legend([plt_legend], loc='upper left') #plt.legend(['train', 'test'], loc='upper left')
pp.savefig()
plt.show()
pp.close()

# save trained weights
model.save_weights(save_name +'.h5')



