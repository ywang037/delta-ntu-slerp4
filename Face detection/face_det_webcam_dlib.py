# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:15:01 2017

@author: wangyuan
"""

# import necessary APIs
import dlib
import cv2
from timeit import default_timer as timer
import math
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

# to crop face in an image
detector = dlib.get_frontal_face_detector()
# uncomment the following line to use cnn face detector
#detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

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
    frame_monitor = frame
   
    # Display the resulting image
    cv2.imshow('Video', frame_monitor)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=sf, fy=sf)    

    # grey scale image is enough for face detection
    grayimage = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
    # detect faces using dlib
    time_face_det_start = timer()
    faces = detector(grayimage, 1)
    time_face_det_end = timer()
    time = str(round(time_face_det_end - time_face_det_start,4))
    
    #print('{} faces are detected in: {} seconds'.format(len(faces),time_face_det_end-time_face_det_start))    
    #print("Number of faces detected: {}".format(len(faces)))    

    # pick the largest face, in dlib.rectangle class
    try:    
        face = max(faces, key=lambda rect: rect.width() * rect.height())
    
        # face locations in small frame
        face_loc_small = [face.left(), face.right(), face.top(), face.bottom()]
        
        # face locations in original frame
        # up_scale = 1/scale_factor
        [left, right, top, bottom] = [math.ceil(x/sf) for x in face_loc_small]   
    	
        # draw a box around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    	
        # display time used by detection	
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, time, (left + 0, bottom + 25), font, 1.0, (255, 255, 255), 1)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        continue
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


