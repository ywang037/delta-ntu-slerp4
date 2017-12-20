# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:15:01 2017

this is a test of face detection performance using opencv

@author: wangyuan
"""

# import necessary APIs
import cv2
import argparse
from timeit import default_timer as timer

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,	help='image file name')
args = vars(ap.parse_args())

# using opencv for face detection
print('Using OpenCV version', cv2.__version__)

# Load the OpenCV face recognition model.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_path = 'test faces/'
img_file = args["image"]
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
    
# detect faces using opencv
time_face_det_start = timer()
faces = face_cascade.detectMultiScale(grayimage, 1.3, 5)
#time_face_det_end = timer()
#print('{} faces are detected in: {} seconds'.format(len(faces),time_face_det_end-time_face_det_start))    

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for (x,y,width,height) in faces:
    cv2.rectangle(image,(x,y),(x+width,y+height),(255,0,0),2)
    roi_gray = grayimage[y:y+height, x:x+width]
    roi_color = image[y:y+height, x:x+width]

time_face_det_end = timer()
print('{} faces are detected in: {} seconds'.format(len(faces),time_face_det_end-time_face_det_start))  

cv2.imshow('imege',image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

## Crop the patch with face detected
#face = faces[0]
#[x1,x2,y1,y2] = [face.left(), face.right(), face.top(), face.bottom()]
#new_image = image[y1:y2, x1:x2]
#new_image = cv2.resize(new_image,(224, 224), interpolation = cv2.INTER_CUBIC)

## Crop the patch with face detected
#for i,d in enumerate(faces): 
#    [x1,x2,y1,y2] = [d.left(), d.right(), d.top(), d.bottom()]
#    new_image = image[y1:y2, x1:x2]
##    new_image = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
#    new_image = cv2.resize(new_image,(224, 224), interpolation = cv2.INTER_CUBIC)
#    win2 = dlib.image_window()
#    win2.set_image(new_image)
#    dlib.hit_enter_to_continue


