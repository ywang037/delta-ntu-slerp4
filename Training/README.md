[This repository](https://github.com/L706077/Face-Recognition-Dataset-for-Training) provides some common datasets for training a cnn.

A common test data set is the [LFW](http://vis-www.cs.umass.edu/lfw/index.html), while it may be too small for training. Using [VGG-Face dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/) for training is reported in this [paper](https://arxiv.org/abs/1710.01494). Another popular training dataset is the [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). The following two projects have been used it:

- [Openface](https://github.com/cmusatyalab/openface)
- [Face recognition using Tensorflow](https://github.com/davidsandberg/facenet)

A washed up CASIA-WebFace data set can be dowoload from [this link](https://pan.baidu.com/s/1kUUP0IN#list/path=%2F). Check out [this webpage](https://github.com/cmusatyalab/openface/issues/119) or [here](https://groups.google.com/forum/#!topic/cmu-openface/Xue_D4_mxDQ) for details.

In addition, the blog ["Building a Facial Recognition Pipeline with Deep Learning in Tensorflow"](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8) is an example tutorial which shows how to use the tensorflow, dlib, and docker. The dlib is used for preprocessing, which could be a good reference.

To train a convnet from stratch, one can adopt any of the following methods:
- Using [retrain.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining) provided by tensorflow official examples, this [blog](https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991) and is a tutorial how to use retrain.py. 
- Following tutorials in [this repo](https://github.com/tensorflow/models/tree/master/research/slim#Training) of tensorflow slim. This directory contains code for training and evaluating several widely used Convolutional Neural Network (CNN) image classification models using TF-slim. It contains scripts that will allow you to train models from scratch or fine-tune them from pre-trained network weights.
- For developing or modifying your own models, see also the main [TF-Slim page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) where training a VGG-16 is provided as an example.
- Using keras function [model.fit](https://keras.io/getting-started/sequential-model-guide/#training), one can refer to section [Fine-tune InceptionV3 on a new set of classes] or [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) as examples how model.fit are invoked.
