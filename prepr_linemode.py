import os
import io
import shutil
from PIL import Image
from natsort import natsorted
import yaml

import prepr_linemode_util as util

raw_data_directory = 'controllercapture_try1'
corrected_ending = '.png'


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
for picture_name in sorted_list_of_files:
    # find mask data
    if picture_name.endswith(mask_ending):
        util.copy_and_rename(raw_data_directory, picture_name, mask_index, mask_path)
        mask_index += 1
        print(picture_name)
    # find depth data
    if picture_name.endswith(depth_ending):
        util.copy_and_rename(raw_data_directory,picture_name,depth_index,depth_path)
        depth_index += 1
        print(picture_name)
    # find depth data
    if picture_name.endswith(rgb_ending) and not any(picture_name.endswith(x) for x in all_endings):
        util.copy_and_rename(raw_data_directory, picture_name, rgb_index, rgb_path)
        rgb_index += 1
        print(picture_name)

# writing the yaml filses:

max_number_of_pics = rgb_index


cam_K = [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
depth_scale = 1.0

i = 0
try:
    os.remove(camera_info_path)
except FileNotFoundError:
    print(camera_info_path + ' doesnt exist')
for i in range(max_number_of_pics):
    util.parse_camera_intrinsic_yml(camera_info_path, i,cam_K,depth_scale)

cam_R_m2c_array = [0.09630630, 0.99404401, 0.05100790, 0.57332098, -0.01350810, -0.81922001, -0.81365103, 0.10814000, -0.57120699]
cam_t_m2c_array = [-105.35775150, -117.52119142, 1014.87701320]
obj_bb = [245,50,23,23]
obj_id = 1

j=0
try:
    os.remove(ground_truth_path)
except FileNotFoundError:
    print(ground_truth_path + ' doesnt exist')
for j in range(max_number_of_pics):
    util.parse_groundtruth_yml(ground_truth_path,j,cam_R_m2c_array,cam_t_m2c_array,obj_bb,16)
