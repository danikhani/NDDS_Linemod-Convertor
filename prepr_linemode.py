import os
import io
import shutil
from PIL import Image
from natsort import natsorted
import yaml
import json
import numpy as np

import prepr_linemode_util as util

raw_data_directory = 'controllercapture_try1'
corrected_ending = '.png'

object_id = 16
# default settings:
cam_K = [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
depth_scale = 1.0
image_size = [640, 480]

#structure of data:
main_data_path = 'generated_linemode/data/16'
main_model_path = 'generated_linemode/models'

depth_folder = 'depth'
mask_folder = 'mask'
rgb_folder = 'rgb'

# yaml files path
ground_truth_path = main_data_path+'/gt.yml'
camera_info_path = main_data_path+'/info.yml'

#what kind of files want to get extraceted
depth_ending = '.depth.mm.16.png'
mask_ending = '.cs.png'
rgb_ending = '.png'
all_endings = ['.cs.png','.16.png','.8.png','.micon.png','.depth.png','.is.png']

# create folder structure
try:
    os.makedirs(main_data_path)
except FileExistsError:
    print(main_data_path + ' exist')
try:
    os.makedirs(main_model_path)
except FileExistsError:
    print(main_model_path + ' exist')
# create folders
depth_path = util.make_empty_folder(main_data_path,depth_folder)
mask_path = util.make_empty_folder(main_data_path,mask_folder)
rgb_path = util.make_empty_folder(main_data_path,rgb_folder)

mask_index = 0
depth_index = 0
rgb_index = 0
sorted_list_of_files = natsorted(os.listdir(raw_data_directory))
for data_name in sorted_list_of_files:
    # find mask data
    if data_name.endswith(mask_ending):
        util.copy_and_rename(raw_data_directory, data_name, mask_index, mask_path)
        mask_index += 1
    # find depth data
    if data_name.endswith(depth_ending):
        util.copy_and_rename(raw_data_directory, data_name, depth_index, depth_path)
        depth_index += 1
    # find rgb data
    if data_name.endswith(rgb_ending) and not any(data_name.endswith(x) for x in all_endings):
        util.copy_and_rename(raw_data_directory, data_name, rgb_index, rgb_path)
        rgb_index += 1
    if data_name.endswith('camera_settings.json'):
        cam_K, depth_scale, image_size = util.get_camera_intrinsic(raw_data_directory,data_name)
        print('camera.settings.json found and instritics updated:')
        print(cam_K)
        print(depth_scale)
        print(image_size)


#removing and recreating the yml files:
try:
    os.remove(ground_truth_path)
except FileNotFoundError:
    print(ground_truth_path + ' doesnt exist')

try:
    os.remove(camera_info_path)
except FileNotFoundError:
    print(camera_info_path + ' doesnt exist')

# writing the yaml filses:
yml_index = 0
for data_name in sorted_list_of_files:
    # writing gt.yml
    if data_name.endswith('.json') and not data_name.endswith('settings.json'):
        #get the arrays from the json files.
        cam_R_m2c_array, cam_t_m2c_array, obj_bb = util.get_groundtruth_data(raw_data_directory, data_name)
        #save arrays in gt.yml
        util.parse_groundtruth_yml(ground_truth_path, yml_index, cam_R_m2c_array, cam_t_m2c_array, obj_bb, object_id)
        # save info.yml
        util.parse_camera_intrinsic_yml(camera_info_path, yml_index, cam_K, depth_scale)
        yml_index +=1

print('data generated!')
print('yml_index,mask_index,depth_index,rgb_index are:')
print(yml_index,mask_index,depth_index,rgb_index)





