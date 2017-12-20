# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:15:01 2017

this is a test of face detection performance using opencv

@author: wangyuan
"""

# import necessary APIs
import cv2
import dlib
import argparse
from timeit import default_timer as timer

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
                required=True,	help='image file name')
ap.add_argument('-d','--detector',
                default='opencv',
                choices=['opencv','dlib'],
                help='face detection engine')
args = vars(ap.parse_args())

img_name = args['image']

'''
# old image path
img_path = 'test faces/'
img_file = args["image"]
#img_file = 'players_1080.jpg'
img_name = img_path + img_file
'''
#img_name = 'test faces/players_1080.jpg'
#img_name = 'test faces/group.jpg'

detector = args['detector']
#detector = 'opencv'
#detector = 'dlib'


def opencv_detection (img_name):
    # using opencv for face detection
    print('Using OpenCV version', cv2.__version__)
    
    # Load the OpenCV face recognition model.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # load image using openCV
    time_load_img_start = timer()
    img_load = cv2.imread(img_name)
    time_load_img_end = timer()
    print('\nImage is loaded in: {} seconds'.format(time_load_img_end-time_load_img_start))
    
    # grey scale image is enough for face detection
    img_gray = cv2.cvtColor(img_load, cv2.COLOR_BGR2GRAY)
    
    # detect faces using opencv
    time_face_det_start = timer()
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5) 
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for (x,y,width,height) in faces:
        cv2.rectangle(img_load,(x,y),(x+width,y+height),(255,0,0),2)
        
    time_face_det_end = timer()
    print('{} faces are detected in: {} seconds'.format(len(faces),time_face_det_end-time_face_det_start))  
    
    cv2.imshow('detected faces',img_load)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def dlib_detection (img_name):
    # using dlib for face detection
    print('Using Dlib')
    
    # hog face detector from dlib
    detector = dlib.get_frontal_face_detector()

    # load image using openCV
    time_load_img_start = timer()
    img_load = cv2.imread(img_name)
    time_load_img_end = timer()
    print('\nImage is loaded in: {} seconds'.format(time_load_img_end-time_load_img_start))
    
    # grey scale image is enough for face detection
    img_gray = cv2.cvtColor(img_load, cv2.COLOR_BGR2GRAY)
    time_face_det_start = timer()
    faces = detector(img_gray, 1)
    for face in faces:    
        cv2.rectangle(img_load, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)    
    
    time_face_det_end = timer()
    print('{} faces are detected in: {} seconds'.format(len(faces),time_face_det_end-time_face_det_start)) 
    cv2.imshow('detected faces',img_load)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
if detector == 'opencv':
    opencv_detection(img_name)
elif detector == 'dlib':
    dlib_detection(img_name)
    
    