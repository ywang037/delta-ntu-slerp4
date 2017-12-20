## General information

In this folder, some [example python programs](./example_face_det_webcam.py) are included to show how to use the [OpenCV](https://opencv.org/) and the [dlib](http://dlib.net/python/index.html) to fulfil face detection. Both of the two libraries can be [installed via anaconda virtual environment](http://www.codesofinterest.com/2016/10/installing-dlib-on-anaconda-python-on.html) conveniently, just like how tensorflow and keras ara installed. 

For installation of OpenCV on Raspberry Pi, one can follow the instructions [in this repository](https://github.com/manashmndl/FabLabRpiWorkshop2017/wiki/Painless-OpenCV-in-Raspbian-Stretch).

Basically, the OpenCV is used for capturing frames from camera stream, loading images from disk, and displaying images on output windows, while the dlib is used for face detection. Some other good eamples for using OpenCV and dlib can be found [in this repository](https://github.com/ageitgey/face_recognition/tree/master/examples) and [in the dlib repository of python examples](https://github.com/davisking/dlib/tree/master/python_examples). 

An interesting comparision of face detection performance [is reported here](https://github.com/andreimuntean/Dlib-vs-OpenCV), it is found that dlib out performs OpenCV in terms of face finding accuracy, however [OpenCV can be much faster than dlib on ARM-based platforms.](https://github.com/cmusatyalab/openface/issues/157)

Indeed, OpenCV face detection is much faster than dlib. However, when it comes to the detection accuracy dlib can detect occluded faces, which cannot be achieved by OpenCV.

## Tips of loading images
To laod images as numpy array, several methods can be adopted
```
"""
Loads an image file (.jpg, .png, etc) into a numpy array
mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
"""
img = scipy.misc.imread(file, mode=mode)
```
or using *scikit-image* which can be installed by `pip install scikit-image` and called as below
```
from skimage import io
img = io.imread(file)
```
or using openCV as
```
import cv2
image = cv2.imread(image_path)
```
