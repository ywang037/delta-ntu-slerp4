'''
on-line test code
mode 1: 'build', build and output feature library, then do feature mapping
mode 2: 'load', load built feature library from save file, then do feature mapping
'''

#import os
#import numpy as np
import pipeline_modules as modules
import argparse
import os
import datetime
import csv

parser = argparse.ArgumentParser()

# command line options of feature library mode
parser.add_argument('-lm',
                    action='store',
                    dest = 'mode',
                    required=True,
                    choices = ['build','load'], 
                    help='Indicate if which mode to select')
# command line argument of path to source image folder or path to the saved feature library
parser.add_argument('-s',
                    action='store',
                    dest='source_path',
                    required=True,
                    help='Specify path to source image folder or path to the saved feature library')
# command line argument of path to input image folder
parser.add_argument('-t',
                    action='store',
                    dest='test_path',
                    required=True,
                    help='Specify path to input test images')
record = parser.parse_args()
mode = record.mode # feature lib mode: 'build_only','bduild_output',or 'load'
source_dir = record.source_path # path to source image folder or path to the saved feature library
test_dir = record.test_path # path to folder of input test images, can also be the value of 'video'

# Hyper parameters of target mobilenet model
mobilenet_input_shape = 224
mobilenet_alpha = 0.5
mobilenet_weights = 'D:/SLERP4_local_wdir/Training/Saved weights/0.5-mobilenet-224-c1771-b64-e200-SGD-lr0.02-20180310-2001.h5'
# Load target mobilenet model
mobilenet_model = modules.cnn_model_initialization(alpha=mobilenet_alpha,
                                                   weights=mobilenet_weights,
                                                   num_class=1771,
                                                   cnn_input_size=mobilenet_input_shape)

# mode 1: build new feature lib from source folder, 
# save the built library, output the built lib into workspace and then do the feature mapping
if mode=='build':
    feature_lib = modules.build_feature_lib(img_source_dir=source_dir,
                                            cnn_model=mobilenet_model,
                                            alpha=mobilenet_alpha,
                                            cnn_input_size=mobilenet_input_shape)
    
# mode 2: load a saved feature lib from file and then do the feature mapping
if mode=='load':
    feature_lib = modules.load_feature_lib(source_dir)

## start recognition by feature mapping 
#feature_lib_data = feature_lib['data'] # load saved feature data

def face_recognition_photos(feature_lib_data,input_dir,cnn_model,alpha,cnn_input_size):
    
    input_abspath=os.path.abspath(input_dir) # set absolute path to the input image folder
    
    # preparing to write test results and timing into record file
    now = datetime.datetime.now()
    test_time=now.strftime("%Y%m%d-%H%M")
    
    record_timing_file = 'timing-'+test_time+'.csv'
    record_results_file = 'results-'+test_time+'.csv'
    
    record_timing=open(record_timing_file,'w')
    record_results=open(record_results_file,'w')
    
    writer_timing=csv.writer(record_timing)
    writer_results=csv.writer(record_results)
    
    writer_timing.writerow(['detection time','extraction time','recognition time'])
    writer_results.writerow(['recognition result','similarity score','ground truth','correctness'])
    
    correctness_stat=0
    total_test_iamges=0
    for subdir in os.listdir(input_abspath):        
        scan_dir = input_abspath+'\\'+ subdir        
        ground_truth = subdir# the name of the subdir is the class name of ground truth           
        for _, _, files, in os.walk(scan_dir):        
            for file in files:        
                # for each input image
                if os.path.splitext(file)[1]=='.jpg':
                    input_image = scan_dir+'\\'+file
                    total_test_iamges+=1
                    # extract new features
                    features_input, time_detection, time_extraction = modules.input_features_from_image(input_image,cnn_model,alpha,cnn_input_size)
                    
                    # for each feature obtained from each input images, do feature mapping
                    time_recognition = 0
                    for feature_input in features_input:
                        recogition_result, similarity_score, rec_time_per_feature=modules.feature_mapping(feature_input,feature_lib_data)
                        if recogition_result==ground_truth:
                            correctness=1 # if recognition is correct, give a score 
                        else:
                            correctness=0
                        correctness_stat+=correctness # accumulate the score
                        # write accuracy related info into file
                        writer_results.writerow([recogition_result,similarity_score,ground_truth,correctness])
                        time_recognition += rec_time_per_feature 
                    
                    # for each input image, write time usage information into file
                    writer_timing.writerow([time_detection,time_extraction,time_recognition])
    print('\nTotal test images = {}'.format(total_test_iamges))
    print('Correct recognition count = {}'.format(correctness_stat))
    print('Accuracy = {:2%}'.format(correctness_stat/total_test_iamges))
    writer_results.writerow(['Total test images','Correct recognition','Accuracy'])
    writer_results.writerow([total_test_iamges,correctness_stat,'{:2%}'.format(correctness_stat/total_test_iamges)])
                    
    # before end of this method, close all opened files
    record_timing.close()
    record_results.close()
    
    
    
# do the face recognition from photos                
face_recognition_photos(feature_lib_data=feature_lib['data'],
                        input_dir=test_dir,
                        cnn_model=mobilenet_model,
                        alpha=mobilenet_alpha,
                        cnn_input_size=mobilenet_input_shape)
#                