[This repository](https://github.com/L706077/Face-Recognition-Dataset-for-Training) provides some common datasets for training a cnn.

A common test data set is the [LFW](http://vis-www.cs.umass.edu/lfw/index.html), while it may be too small for training. Using [VGG-Face dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/) for training is reported in this [paper](https://arxiv.org/abs/1710.01494). Another popular training dataset is the [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). The following two projects have been used it:

- [Openface](https://github.com/cmusatyalab/openface)
- [Face recognition using Tensorflow](https://github.com/davidsandberg/facenet)

A washed up CASIA-WebFace data set can be dowoload from [this link](https://pan.baidu.com/s/1kUUP0IN#list/path=%2F). Check out [this webpage](https://github.com/cmusatyalab/openface/issues/119) or [here](https://groups.google.com/forum/#!topic/cmu-openface/Xue_D4_mxDQ) for details.

In addition, the blog [Building a Facial Recognition Pipeline with Deep Learning in Tensorflow](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8) is an example tutorial which shows how to use the tensorflow, dlib, and docker. The dlib is used for preprocessing, which could be a good reference.

To train a convnet from stratch, one can adopt any of the following methods:

- Following tutorials in [this repository of tensorflow slim](https://github.com/tensorflow/models/tree/master/research/slim#Training) which contains code for training and evaluating several widely used Convolutional Neural Network (CNN) image classification models using TF-slim. It contains scripts that will allow you to train models from scratch or fine-tune them from pre-trained network weights.
- For developing or modifying your own models, see also the [main TF-Slim page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) where training a VGG-16 is provided as an example.
- Using keras function [model.fit](https://keras.io/getting-started/sequential-model-guide/#training), one can refer to section [Fine-tune InceptionV3 on a new set of classes](https://keras.io/applications/) or [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) as examples how model.fit are invoked. Some additional tutorials on training a cnn using keras can be found below:
  - [Keras Tutorial: The Ultimate Beginner’s Guide to Deep Learning in Python](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)
  - [Develop Your First Neural Network in Python With Keras Step-By-Step](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
  - [Training neural networks efficiently using Keras](https://www.packtpub.com/books/content/training-neural-networks-efficiently-using-keras)
  - [Training AlexNet, using Keras and Theano (the most useful example)](https://github.com/duggalrahul/AlexNet-Experiments-Keras/blob/master/Code/AlexNet_Experiments.ipynb)

The following shows how to take a pre-trained Inception v3 or Mobilenet model, and train a new top layer that can recognize other classes of images (transfer learning).
- Using [retrain.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining) provided by tensorflow official examples, this [blog](https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991) and [this one](https://hackernoon.com/building-an-insanely-fast-image-classifier-on-android-with-mobilenets-in-tensorflow-dc3e0c4410d4) are tutorials which how to use retrain.py as couples of lines of codes in python.

Besides, [this webpage](https://github.com/Zehaos/MobileNet/issues/33) discusses roughly how to choose hyperparameters for training a MobileNet, the scripts for training is [here](https://github.com/Zehaos/MobileNet/blob/master/train_image_classifier.py) that looks quite similar to the codes in the second bullet above.

The official github repository of MobileNets can be found [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
