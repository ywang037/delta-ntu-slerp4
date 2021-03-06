# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:15:01 2017

@author: wangyuan
"""

# import necessary APIs
import dlib
import cv2
from timeit import default_timer as timer



# to crop face in an image
detector = dlib.get_frontal_face_detector()
# uncomment the following line to use cnn face detector
#detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# This is a GUI window capable of showing images on the screen.
win = dlib.image_window()

img_path = 'test faces/'
img_file = 'p1_1080.jpg'
#img_file = 'p2_720.jpg'
#img_file = 'p3_720.jpg'
#img_file = 'P2_480.jpg'
img_name = img_path + img_file

# load image using openCV
time_load_img_start = timer()
image = cv2.imread(img_name)
time_load_img_end = timer()
print('\nImage is loaded in: {} seconds'.format(time_load_img_end-time_load_img_start))

# grey scale image is enough for face detection
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# detect faces using dlib
time_face_det_start = timer()
faces = detector(grayimage, 1)
time_face_det_end = timer()
print('{} faces are detected in: {} seconds'.format(len(faces),time_face_det_end-time_face_det_start))    
#print("Number of faces detected: {}".format(len(faces)))    

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
win.clear_overlay()
win.set_image(image)
win.add_overlay(faces)
#dlib.hit_enter_to_continue()

## Crop the patch with face detected
#face = faces[0]
#[x1,x2,y1,y2] = [face.left(), face.right(), face.top(), face.bottom()]
#new_image = image[y1:y2, x1:x2]
#new_image = cv2.resize(new_image,(224, 224), interpolation = cv2.INTER_CUBIC)

# Crop the patch with face detected
for i,d in enumerate(faces): 
    [x1,x2,y1,y2] = [d.left(), d.right(), d.top(), d.bottom()]
    new_image = image[y1:y2, x1:x2]
#    new_image = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    new_image = cv2.resize(new_image,(224, 224), interpolation = cv2.INTER_CUBIC)

win2 = dlib.image_window()
win2.set_image(new_image)
dlib.hit_enter_to_continue


