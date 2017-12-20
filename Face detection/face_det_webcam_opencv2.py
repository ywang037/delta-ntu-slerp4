# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:15:01 2017

@author: wangyuan
"""

# import necessary APIs
import cv2
from timeit import default_timer as timer
#import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-r', '--resolution', 
                default='720p', 
                choices=['480p','720p','1080p'],	
                help='camera resolution')
ap.add_argument('-sf','--scalefactor',
                default=1.0,
                type=float,
                help='multiplication factor for resizing frame dimension')
args = vars(ap.parse_args())

# using opencv for face detection
print('Using OpenCV version', cv2.__version__)

# Load the OpenCV face recognition model.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# initiate camera
video_capture = cv2.VideoCapture(0)
cam_res=args['resolution']
if cam_res == '1080p':
    # set resolution to 1080p
    video_capture.set(3, 1920.)
    video_capture.set(4, 1080.)
elif cam_res == '720p':
    # set resolution to 720p
    video_capture.set(3, 1280.)
    video_capture.set(4, 720.)
elif cam_res =='480p':
    # set resolution to 480p
    video_capture.set(3, 640.)
    video_capture.set(4, 480.)

# scaling factor for detection 
sf = args['scalefactor'];

while True:

    ret, frame = video_capture.read()
#    print('The frame size is {}'.format(frame.shape))

       
#    # Display the resulting image
#    frame_monitor = frame
#    cv2.imshow('Video', frame_monitor)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=sf, fy=sf)    

    # grey scale image is enough for face detection
    grayimage = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
    # detect faces using dlib
    time_face_det_start = timer()
    faces = face_cascade.detectMultiScale(grayimage, 1.3, 5)
    time_face_det_end = timer()
    time = str(round(time_face_det_end - time_face_det_start,4))
    
    #print('{} faces are detected in: {} seconds'.format(len(faces),time_face_det_end-time_face_det_start))    
    #print("Number of faces detected: {}".format(len(faces)))    

    font = cv2.FONT_HERSHEY_DUPLEX # set font of screen display
    msg_1 = '{} faces detected'.format(len(faces))    
    cv2.putText(frame, msg_1, (20, 30), font, 1.0, (0,63,255), 1)
    if not str(msg_1):
        msg_2 = 'Time taken: {} seconds'.format(time)
        cv2.putText(frame, msg_2, (20, 60), font, 1.0, (0,63,255), 1)
        
    id=0
    for (x,y,width,height) in faces:
        cv2.rectangle(frame,(x,y),(x+width,y+height),(255,0,0),2)
        id+=1
        cv2.putText(frame, 'ID: '+str(id), (x+0, y+height+25), font, 1.0, (0,63,255), 1)           
    
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


