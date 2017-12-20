# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:15:01 2017

@author: wangyuan
"""

# import necessary APIs
import cv2
from timeit import default_timer as timer
import math
import argparse
import datetime

ap = argparse.ArgumentParser()
ap.add_argument('-r', '--resolution', 
                default='720p', 
                choices=['480p','720p','1080p'],	
                help='camera resolution')
ap.add_argument('-sf','--scalefactor',
                default=1.0,
                type=float,
                help='multiplication factor for resizing frame dimension')
ap.add_argument('-bm','--boxmultiplier',
                default=2.0,
                type=float,
                help='multiplier used to amplify the face bounding box')
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

# set font of screen display
font = cv2.FONT_HERSHEY_DUPLEX

# box multiplier used to amplify original area of detected faces
# in order to get a larger bounding box of faces
bx_mp = args['boxmultiplier']

while True:
    ret, frame = video_capture.read()

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
    
    # number of detected faces
    nf = len(faces)
    
    # get current date and time
    now = datetime.datetime.now()       
    
    if nf:
        msg_1 = '{} faces detected at '.format(nf)+now.strftime('%Y-%m-%d %H:%M:%S')
        print(msg_1+' in {} seconds'.format(time)) # output to console
        cv2.putText(frame, msg_1, (20,30), font, 1.0, (0,63,255), 1)
        msg_2 = 'Time taken: {} seconds'.format(time)
        cv2.putText(frame, msg_2, (20,60), font, 1.0, (0,63,255), 1)
    else:
        msg_3 = 'No face detected'
        print(msg_3) # output to console
        cv2.putText(frame, msg_3, (20,30), font, 1.0, (0,63,255), 1)
        
    id=0 # re-initialize numb of id to zero
    for (x,y,width,height) in faces:
        # scale back to normal size
        face_loc_small = [x,y,width,height]
        
        # face location coordinates in normal size frame
        [x_norm, y_norm, width_norm, height_norm] = [math.ceil(z/sf) for z in face_loc_small]
        
        # face bounding box covers a region slightly larger than the face,
        # use bx_mp to control the percentage 
        bx_left = math.ceil(x_norm + 0.5*(1-bx_mp)*width_norm)
        bx_top = math.ceil(y_norm + 0.5*(1-bx_mp)*height_norm)
        bx_right = math.ceil(x_norm + 0.5*(1+bx_mp)*width_norm)
        bx_bottom = math.ceil(y_norm + 0.5*(1+bx_mp)*height_norm)
        bx_width = math.ceil(bx_mp*width_norm)
        bx_height = math.ceil(bx_mp*height_norm)
        
        # draw a box for detected faces
        cv2.rectangle(frame,(bx_left,bx_top),(bx_right, bx_bottom),(255,0,0),2)
        
        # output the size of detected faces to console
        msg_4 = 'The size of detected face is: {}-by-{}'.format(bx_width,bx_height)
        print(msg_4)
        
        # display the size of detected faces on screen
        id += 1
        cv2.putText(frame, 'ID '+str(id)+': ({},{})'.format(bx_width,bx_height), 
                    (bx_left, bx_bottom+25), font, 1.0, (0,63,255), 1)
    
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


