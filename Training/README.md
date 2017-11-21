[This repository](https://github.com/L706077/Face-Recognition-Dataset-for-Training) provides some common datasets for training a cnn.

A common test data set is the [LFW](http://vis-www.cs.umass.edu/lfw/index.html), while it may be too small for training. Using [VGG-Face dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/) for training is reported in this [paper](https://arxiv.org/abs/1710.01494). Another popular training dataset is the [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). The following two projects have been used it:

- [Openface](https://github.com/cmusatyalab/openface)
- [Face recognition using Tensorflow](https://github.com/davidsandberg/facenet)

A washed up CASIA-WebFace data set can be dowoload from [this link](https://pan.baidu.com/s/1kUUP0IN#list/path=%2F). Check out [this webpage](https://github.com/cmusatyalab/openface/issues/119) or [here](https://groups.google.com/forum/#!topic/cmu-openface/Xue_D4_mxDQ) for details.

The blog ["Building a Facial Recognition Pipeline with Deep Learning in Tensorflow"](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8) is an example tutorial which shows how to use the tensorflow, dlib, and docker. The dlib is used for preprocessing, which could be a good reference.
