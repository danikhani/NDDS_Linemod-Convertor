import os
import shutil
from PIL import Image
from natsort import natsorted

import prepr_linemode_util as util

raw_data_directory = 'controllercapture_try1'
corrected_ending = '.png'


#structure of data:
main_data_path = 'generated_linemode/data/16'
main_model_path = 'generated_linemode/models'

depth_folder = 'depth'
mask_folder = 'mask'
rgb_folder = 'rgb'

#what kind of files want to get extraceted
depth_ending = '.depth.cm.16.png'
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
