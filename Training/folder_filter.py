# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:16:48 2017

@author: SLERP4

delete subfolders that contains files less than certain threshold
"""

import os 
import shutil
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', 
                default=os.getcwd(),
                help='path to the folder contains subfolders where files are storesd')
ap.add_argument('-t', '--type',
                default='.jpg',
                help='type of the files needs to count')
ap.add_argument('-th', '--threshold',
                default=100,
                type=int,
                help='minimum number of files to filter')
ap.add_argument('-m', '--mode',
                default='scan',
                choices=['scan', 'filter'],
                help='to scan only (no removal) or execute filter directly')
ap.add_argument('-sm', '--silentmode',
                default=False,
                choices=['True', 'False'])
args = vars(ap.parse_args())

os.chdir(args['path']) # change working dir to input 'path'
wdir = os.getcwd()
#wdir = args['path']
file_type = args['type']
thld = args['threshold']
mode = args['mode']
silent = args['silentmode']

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
#                print(file)
    return count       
            

def scan (wdir,file_type,thld,silent):
#    num_dir = len(os.listdir(wdir)) # original number of items (folders or files) in path 
    num_dir_found = 0 # number of folder found in path
    num_dir_rm=0 # number of fodler to be removed
    count_total_found = 0 # total number of files found
    count_total_rm = 0    # total number of files to be removed
    for sub_dir in os.listdir(wdir):  
        if os.path.isdir(sub_dir):
            # if sub_dir is found to be a folder
            num_dir_found += 1 
            count = file_count(sub_dir,file_type)
            count_total_found += count
            if count < thld:                
                num_dir_rm += 1
                count_total_rm += count
                if not silent:
                    print(wdir+'\\'+sub_dir+' will be be removed')

    if num_dir_found == 0:
        print('No folder found in '+ wdir)
    else:
        print('\n{} folders found, \t{} will be removed'.format(num_dir_found,num_dir_rm))
        print('{} files found, \t{} will be removed'.format(count_total_found,count_total_rm))
        print('{} files retained, \t{} folders retatined'.format(count_total_found-count_total_rm,num_dir_found-num_dir_rm))

        
def ffilter (wdir,file_type,thld,silent):
#    num_dir = len(os.listdir(wdir)) # original number of items (folders or files) in path 
    num_dir_found = 0 # number of folder found in path
    num_dir_rm=0 # number of fodler to be removed
    count_total_found = 0 # total number of files found
    count_total_rm = 0    # total number of files to be removed
    for sub_dir in os.listdir(wdir):  
        if os.path.isdir(sub_dir):
            # if sub_dir is found to be a folder
            num_dir_found += 1 
            count = file_count(sub_dir,file_type)
            count_total_found += count
            if count < thld:                
                # folders contains file less than threshold will be removed from input 'path'
                shutil.rmtree(sub_dir)
                num_dir_rm += 1
                count_total_rm += count
                if not silent:
                    print(args['path']+'\\'+sub_dir+' has been removed')
                    
    if num_dir_found == 0:
        print('No folder found in '+ wdir)
    else:
        print('\n{:>15}'.format('Dir total')+'\t{:>15}'.format('Dir removed')
        +'\t{:>15}'.format('File total')+'\t{:>15}'.format('File removed'))
#        print('\nDir removed'+'\tDir total'+'\tFile removed'+'\tFile total')
        print('{:>15d}'.format(num_dir_rm)+'\t{:>15d}'.format(num_dir_found)
        +'\t{:>15d}'.format(count_total_rm)+'\t{:>15d}'.format(count_total_found))
        
if mode=='scan':
    scan(wdir,file_type,thld,silent)
elif mode=='filter':
    ffilter(wdir,file_type,thld,silent)