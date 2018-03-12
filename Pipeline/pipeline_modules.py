# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:48:12 2018

@author: wangyuan
"""
#import os, importlib
import os
import cv2
import datetime
import numpy as np
import math
from PIL import Image as image_pil
from timeit import default_timer as timer
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image as image_k
from keras.models import Model
#from keras import backend as K
#import json

def cnn_model_initialization(alpha, weights, num_class, cnn_input_size=224):
    # this function is to be invoked once at the start of the pipeline
    '''
    # check and set backend
    tf = 'tensorflow'
    if K.backend() != tf:
        os.environ['KERAS_BACKEND'] = tf
        importlib.reload(K)
        assert K.backend() == tf
        print('{} backend is sucessfully set'.format(K.backend()))
    elif K.backend() == tf:
        print('{} backend has already been set'.format(K.backend()))
    '''
    # Setup the model
    time_load_model_start = timer()
    size=cnn_input_size
    # load the base model with top layer, since your weights are trained using a topped model
    base_model = MobileNet(input_shape=(size,size,3), 
                      alpha=alpha, 
                      depth_multiplier=1, 
                      dropout=1e-3, 
                      include_top=True, 
                      weights=weights, 
                      input_tensor=None, 
                      pooling=None, 
                      classes=num_class)
    # define a new model whose output is the reshape layer of the base model
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('reshape_1').output)
    #model.summary()
    time_load_model_end = timer()
    time_load_model = str(round(time_load_model_end-time_load_model_start,4))
    print('Deep CNN model Mobilenet is loaded, time taken: {} seconds'.format(time_load_model))
    return model

def face_detection_initializaiton():
    # this function is only invoked once in the pipline
    print('Face detector is loaded')
    print('Using OpenCV version', cv2.__version__) 
    # load face detector
    face_detector = cv2.CascadeClassifier('Face detection/haarcascade_frontalface_default.xml')
    return face_detector

def load_images(img_path):
    '''
    img_path: input image path, type=string
    '''
    time_load_img_start = timer()
    img_loaded = cv2.imread(img_path) # in BGR format
    time_load_img_end = timer()
    #time_load_img = time_load_img_end - time_load_img_start
    time_load_img = str(round(time_load_img_end - time_load_img_start,5)) # 4-digits preservation for displaying loading time
    print('\nImage {} is loaded in: {} seconds'.format(img_path,time_load_img)) # display time taken for loading per image
    return img_loaded

def video_streaming():
    video_capture = cv2.VideoCapture(0)
    # set resolution to 1080p
    video_capture.set(3, 1920.)
    video_capture.set(4, 1080.)
    ret, frame = video_capture.read()
    return frame    
    
def face_detection(img_input,face_detector):
    '''    
    img_input: 
        either the output of load_images(img_path), type=ndarray, BGR format
        or the output of video_streaming(), type = ndarray, BGR format
    face_detector: output of face_detection_initializaiton()
    the output of this funciton is the cooridnates of detected faces
    
    '''        
    # grey scale image is enough for face detection
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)    
        
    # detect faces using opencv
    time_face_det_start = timer()
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    time_face_det_end = timer()
    time_face_det = round(time_face_det_end - time_face_det_start,5) # 4-digits preservation for displaying detection time
    
    # number of detected faces
    num_faces = len(faces)
    if num_faces:
        print('{} faces are detected in: {} seconds'.format(num_faces,time_face_det))
    else:
        print('WARNING: No face detected')
            
    return faces, num_faces, time_face_det

def feature_extraction (img_loaded, detected_faces, model, alpha, cnn_input_size=224):
    '''
    img_loaded: is the output of load_images(img_path), type=ndarray, BGR format
    detected faces: output of face_detection(img_loaded,face_detector), type = ndarray
    model: the trained/selected cnn model for feature extraction, 
           it is the output of cnn_model_initialization(shape_input, alpha, weights):
    size_cnn_input: the size of input image to target cnn model for feature extraction
    '''
    size = cnn_input_size
    feature_dim = int(alpha*1024)
    # face_features = np.empty_like(feature)
    face_features = np.empty([1,feature_dim])
    time_extract = 0
    for (x,y,width,height) in detected_faces:
        # step 1: cropp face patch of detected face from input image        
        # find coordinates/bounding box of detected face
        [left, right, top, bottom] = [x, x+width, y, y+height]         
        # crop the face patch from the input image and convert into RGB colors
        face_patch = img_loaded[top:bottom, left:right]
        # color conversion maybe redundant for feature extraction, you can try remove it and see the difference        
        face_patch_rgb = cv2.cvtColor(face_patch, cv2.COLOR_BGR2RGB)        
        # interpolation, resize to the standard input size of cnn for feature extraction
        face_patch_rgb_std_cv = cv2.resize(face_patch_rgb,(size, size), interpolation = cv2.INTER_CUBIC)
        # convert face patch from openCV format to PIL format, this requires PIL pre-installed
        face_patch_rgb_std_pil = image_pil.fromarray(face_patch_rgb_std_cv)
        
        # step 2: extract features from cropped face patch
        time_extract_start = timer()
        # Converts a PIL Image instance to a N-dimensional Numpy array.
        face_patch_input = image_k.img_to_array(face_patch_rgb_std_pil)
        # increase the dimension of the Numpy array to (1,224,224,3)
        face_patch_input = np.expand_dims(face_patch_input, axis=0)
        face_patch_input = preprocess_input(face_patch_input)   
        # extract feature using cnn model
        feature = model.predict(face_patch_input)
        feature = np.reshape(feature,(1,feature_dim))
        time_extract_end = timer()
        time_extract += round(time_extract_end-time_extract_start,5)
        # print results on screen
        print('Feature of dimension {} are extracted from last pooling layer, using {} seconds.'.format(feature.shape,time_extract))
        
        # save the feature of all detected faces into an numpy array called features (similar to faces) for further action
        face_features = np.vstack((face_features,feature))
    
    # the M-dimensional numpy array that stores all the extracted features form an input image
    # the M equals to the number of face appeared in the input image
    face_features = face_features[1:,:] 
    
    return face_features, time_extract

def build_feature_lib(img_source_dir,cnn_model,alpha,cnn_input_size=224):
    '''
    Build the feature libraries for individuals of interested
    img_source_dir: the directory of images from where the feature libraries will be built
    
    ''' 
    now = datetime.datetime.now() #current date and time
    lib_save_name = 'feature-lib-built-'+now.strftime("%Y%m%d-%H%M")
    #stat_save_name= 'feature_stat'+img_source_dir+'.json'
    
    # initialize the dictionary for saving feature data and stat info
    feature_lib = {}
    feature_lib['source']=img_source_dir    
    feature_lib['saved file']=lib_save_name
    feature_lib_data = {}
    feature_num_stat = {}
    
    # load face detector 
    face_detector= face_detection_initializaiton()
    feature_dim = int(alpha*1024)
    
    id_num = 1
    for subdir in os.listdir(img_source_dir):
        source_abspath = os.path.abspath(img_source_dir)
        scan_dir = source_abspath+'\\'+ subdir
        print('Scanning {} for {}-th individuals'.format(scan_dir,id_num))
        features_per_id=np.empty([1,feature_dim])    
        count_feature = 0
        for roots, dirs, files, in os.walk(scan_dir):  
            for file in files:        
                # extract features from each input image
                if os.path.splitext(file)[1]=='.jpg':
                    image_path = scan_dir+'\\'+file
                    img_loaded = load_images(image_path)                    
                    # face detection to crop faces for feature extraction
                    detected_faces,n_feature,_ = face_detection(img_loaded,face_detector)
                    count_feature += n_feature
                    # feature extraction using cnn model
                    features,_ = feature_extraction(img_loaded,detected_faces,cnn_model,alpha,cnn_input_size)
                    features_per_id=np.vstack((features_per_id,features))
                        
        features_per_id = features_per_id[1:,:]        
        lib_per_id = 'feature_'+subdir
        np.save(lib_per_id,features_per_id)  
        feature_lib_data[subdir]=features_per_id
        feature_num_stat[subdir]=count_feature
        id_num+=1
        
    # include the feature_lib_data dictionary as a nested dic in the dictionary feature_lib   
    feature_lib['data']=feature_lib_data
    # include the feature_num_stat dictionary as a nested dic in the dictionary feature_lib   
    feature_lib['stat']=feature_num_stat
    # save the dictionary feature_lib as a npy file
    np.save(lib_save_name,feature_lib) 
    '''
    # save the statistics of the built feature lib into a json file
    with open(stat_save_name,'w') as save_dic:
        json.dump(feature_lib_stat,save_dic)
    print('\nDone! Feature libraries has been built. ')
    '''
    return feature_lib


def load_feature_lib(feature_lib_name):
    # load saved feature lib which is a npy file
    feature_lib_load = np.load(feature_lib_name)
    feature_lib = feature_lib_load.item()
    '''
    # load saved feature lib stat info which is a json file
    with open(feature_lib_stat_name) as open_lib_stat:
        feature_lib_stat = json.load(open_lib_stat)
    '''
    return feature_lib
        

def input_features_from_image(input_image,cnn_model,alpha,cnn_input_size):
    '''
    input_image: path to an input image
    cnn_model: trained mobilenet model
    alpha: hyper-parameter alpha used in the trained mobilenet model
    cnn_input_size: hyper-parameter input shape of the trained mobilenet model
    '''
    img_loaded=load_images(input_image) # load image as cv2 format    
    face_detector= face_detection_initializaiton() # load face detector 
    detected_faces,_,detection_time = face_detection(img_loaded,face_detector) # detect and crop faces from input image   
    features, extraction_time = feature_extraction(img_loaded,detected_faces,cnn_model,alpha,cnn_input_size)
    return features, detection_time, extraction_time
        
def input_features_from_video(input_frame,cnn_model,alpha,cnn_input_size):
    '''
    input_frame is the output of video_streaming(), in cv2 format
    '''
    face_detector= face_detection_initializaiton() # load face detector 
    detected_faces,_,detection_time = face_detection(input_frame,face_detector)
    features, extraction_time = feature_extraction(input_frame,detected_faces,cnn_model,alpha,cnn_input_size)
    return features, detection_time, extraction_time

def feature_mapping (feature_input,feature_lib):
    '''
    feature_input is a feature vector extracted in real time, a row vector already
    feature_lib is the built or loaded feature libraries, dictionary type, feature_lib['data']
    '''
    time_mapping_start=timer()
    beta_lb = 0   # lower bound of beta in softmax function
    beta_ub = 21  # upper bound of beta in softmax function
    similarity_score = []  # mean value of similarities between captured feature and stored features of each identity    
    class_names = sorted(feature_lib.keys()) # class names in the feature lib
    
    # for each identity in the feature library
    for identity in class_names:
        # similarity score for this identity under a specific parameter beta
        sim_score_beta = []  
        for beta in range(beta_lb,beta_ub):
            angle = []  # list of cosine angles, len(angle)= number of features of this identity
            angle_exp = []  # list of exponential cosine angles
            
            # all features of an identity, a matrix of n-by-1024, n is number of features of this identity
            features_identity = feature_lib[identity] 
            
            # calculate the cosine angle between two feature vectors       
            for feature in features_identity:
                feature = np.expand_dims(feature,1) # make a column vector of 1024-by-1
                num = np.dot(feature_input,feature).item()
                denom = np.linalg.norm(feature) * np.linalg.norm(feature_input)
                # cosine angle between the input feature and each feature of this particular identity
                angle_new = num/denom 
                # calculate the exponential similarity
                angle_new_exp = math.exp(beta*angle_new)
                angle.append(angle_new)
                angle_exp.append(angle_new_exp)
                
            # softmax function of similarity under a specific beta
            softmax_new = np.dot(angle,angle_exp)/np.linalg.norm(np.array(angle_exp),1)
            sim_score_beta.append(softmax_new)
        
        # for this idendity, 
        # find mean value of similarities over different values of betas, and store in a list, i.e., vector 
        similarity_score.append(np.mean(sim_score_beta))  
 
    result_identity_index = np.array(similarity_score).argmax()  # convert the variable type from 'list' to 'array'
    #result_identity_index = similarity_score.argmax()  # recoginition result of the captured feature    
    recogition_result = class_names[result_identity_index] # find the name of the resulting identity, string type
    time_mapping_end=timer()
    time_mapping = round(time_mapping_end-time_mapping_start,5)
    print('Class = {}, similarity scores = {}'.format(recogition_result,similarity_score))
    return recogition_result, similarity_score, time_mapping

#for key in sorted(dic.keys()):
#    print(key)