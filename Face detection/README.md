## General information

This folder collects information and tutorials from internet that introduce how to do face detection, using the [dlib](http://dlib.net/python/index.html) or [OpenCV](https://opencv.org/). 

The project [OpenFace](https://cmusatyalab.github.io/openface/) introduce how they implement the entire pipline with FaceNet.

The blog [Building a Facial Recognition Pipeline with Deep Learning in Tensorflow](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8) also uses dlib.

This page [Real-time facial landmark detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/) and other related articles e.g., [How to install dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) on the same websites (just use search bar with keyword dlib or opencv) are good tutorials to kick off.

## Loading images
To laod images as numpy array, several methods can be adopted
```
"""
Loads an image file (.jpg, .png, etc) into a numpy array
mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
"""
img = scipy.misc.imread(file, mode=mode)
```
or using *scikit-image* which can be installed via the command:
```
pip install scikit-image
```
and call like below
```
from skimage import io
img = io.imread(file)
```
or using openCV as
```
import cv2
image = cv2.imread(image_path)
```
