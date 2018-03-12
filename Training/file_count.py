# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:00:41 2017

count number of files in each subfolder

@author: SLERP4
"""
import os 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', 
                default=os.getcwd(),
                help='path to the folder contains subfolders where files are storesd')
ap.add_argument('-t', '--type',
                default='.jpg',
                help='type of the files needs to count')
args = vars(ap.parse_args())


def file_count (file_path,file_type):
    '''
    file_path = the folder where certain type of file are actually counted
    file_type = the type of file which is you want to count
    '''
    count=0
    for roots, dirs, files, in os.walk(file_path):
        for file in files:
            if os.path.splitext(file)[1]==file_type:
                count += 1            
    return count       
            
os.chdir(args['path']) # change working dir to 'path'
wdir = os.getcwd()
file_type = args['type']

print('Number'+'\tType'+'\tDirectory')

for sub_dir in os.listdir(wdir):
    if os.path.isdir(sub_dir):
        img_count = file_count(sub_dir,file_type)
        print('{} '.format(img_count)+'\t'+args['type']+ '\t'+args['path']+'\\'+sub_dir) 
#    print('{} '.format(img_count)+'\t'+args['type']+ ' files found in '+args['path']+'\\'+sub_dir)
 
